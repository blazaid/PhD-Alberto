import collections

import numpy as np
import tensorflow as tf

IVar = collections.namedtuple('IVar', ('name', 'fuzzy_sets', 'domain'))
OVar = collections.namedtuple('OVar', ('name', 'values'))


def slope_asc(x, a, db):
    """ Tensor with the operation of an ascendent line.

    :param x: A tensor with the values to apply the function.
    :param a: The position where the lines stops being 0.
    :param db: The distance from the previous point (a) to the position where
        starts being 1.
    :return: A tensor of the same shape with the values of applying this
        function to the x tensor.
    """
    return tf.minimum(tf.maximum((x - a) / db, 0), 1)


def slope_desc(x, a, db):
    """ Tensor with the operation of an descendent line.

    :param x: A tensor with the values to apply the function.
    :param a: The position where the lines stops being 0.
    :param db: The distance from the previous point(a) to the position where
        starts being 1.
    :return: A tensor of the same shape with the values of applying this
        function to the x tensor.
    """
    return 1 - slope_asc(x, a, db)


def trapezoid(x, a, db, dc, dd):
    """ Returns a tensor with the operation of a trapezoidal mf.
    
    This operation will be a composition of two slopes like the ones defined
    in function `line`. This means that it is necessary to initialize it with
    proper slopes values (i.e. first one with a positive value and the second
    one with a negative value).
    
    :param x: A tensor with the values to apply the function.
    :param a: The first (leftmost) point of the trapezoid.
    :param db: Distance from previous point (a) to the point where the
        trapezoid starts being 1 (point b). The function transforms it to a
        positive number so it is always greater or equal to 0.
    :param dc: Distance from previous point (b) to the point where the
        trapezoid stops being 1 (point c). The function transforms it to a
        positive number so it is always greater or equal to 0.
    :param dd: Distance from previous point (c) to the point where the
        trapezoid starts being 0 (point d). The function transforms it to a
        positive number so it is always greater or equal to 0.
    :return: A tensor of the same shape with the values of applying this
        function to the x tensor.
    """
    line_asc = (x - a) / db
    line_des = (a + db + dc - x) / dd + 1
    union = tf.minimum(line_asc, line_des)
    return tf.minimum(tf.maximum(union, 0), 1)


def fuzzification_graph(x, var_desc):
    """ TBD

    :param x: A tensor of shape (m, 1) wher m is each of the example values.
    :param var_desc: The description of this variable as a `IVar` tuple.
    :return: A tensor of shape (m, n) where m is the number of examples and n
        the number of fuzzy sets in this partition. The values will be the
        result of the fuzzification process where each column corresponds to
        this fuzzy set's membership function to the value.
    """
    with tf.variable_scope(var_desc.name):
        # The variables of this graph will be the shifts between points of the
        # membership functions. This shifts will be initialized to unfold all
        # the points equidistantly
        num_points = (var_desc.fuzzy_sets - 1) * 2
        lo, hi = min(var_desc.domain), max(var_desc.domain)
        shift_size = (hi - lo) / (num_points + 1)
        shifts = [
            tf.abs(tf.Variable(shift_size, name='s{}'.format(i)))
            for i in range(num_points)
        ]

        # We define where the domain starts to define the rest of elements
        # via shifts
        base = tf.constant(name='b', value=lo, dtype=tf.float32)

        # Now, we create all the sets
        next_fs_starting_point = base + shifts[0]
        fuzzy_sets = []
        for i in range(var_desc.fuzzy_sets):
            # Depending on the index, create either desc, asc or trap fs.
            if i == 0:
                # First fuzzy set should be a descendent line
                fs = slope_desc(x, next_fs_starting_point, shifts[1])
            elif i == var_desc.fuzzy_sets - 1:
                # Last fuzzy set should be an ascendent line
                fs = slope_asc(x, next_fs_starting_point, shifts[-1])
            else:
                # Inner fuzzy sets should be a trapezoids
                shifts_to_apply = shifts[i * 2 - 1:i * 2 + 2]
                fs = trapezoid(x, next_fs_starting_point, *shifts_to_apply)
                next_fs_starting_point += shifts[(i - 1) * 2 + 1] + shifts[(i - 1) * 2 + 2]

            # Add this fs to the list of fuzzy_sets
            fuzzy_sets.append(fs)

        # Now concat all the fuzzy sets
        all_fuzzy_sets = tf.concat(fuzzy_sets, 0)

        # Return the created variable and the tensor with the fuzzifications
        # of the inputs
        return tf.transpose(tf.reshape(all_fuzzy_sets, (var_desc.fuzzy_sets, -1)))


def inference_graph(fuzzy_inputs, num_fuzzy_inputs, num_fuzzy_outputs):
    """ Creates the subgraph related to the fuzzy rules.
    
    :param f_inputs:
    :param num_f_outputs: The value of each singleton for each fuzzy output.
    """
    # First we create the cartesian product between all the fuzzy values of
    # each variable and then reduce them with the t-norm of all the elements
    # and make the t-norm along the resulting elements (we call'em inferences).
    #
    # In the end inference will be the minimum of each combination between
    # fuzzy inputs, where the rows are the examples and the columns each
    # combination of fuzzy inputs.
    m = tf.shape(fuzzy_inputs[0])[0]
    inference = fuzzy_inputs[0]
    for g in fuzzy_inputs[1:]:
        inference = tf.minimum(inference[:, None], g[:, :, None])
        inference = tf.reshape(inference, (m, -1))

    # Then, we create a set of weights of the size of the inference times the
    # number of fuzzy outputs. This implies that each of the inference will
    # have a weight over the final result. It can be change to binary values
    # to denote "this inference has/hasn't to do with this fuzzy output.
    num_combinations = np.prod(num_fuzzy_inputs)
    fuzzy_output_weights = tf.get_variable(
        'fuzzy_output_weights',
        shape=[num_fuzzy_outputs, 1, num_combinations],
        initializer=tf.ones_initializer(),
    )
    inference = tf.multiply(inference, tf.sigmoid(fuzzy_output_weights))

    # Now we reduce to the max the values of each of the outputs
    return tf.transpose(tf.reduce_max(inference, axis=2))


def defuzzification_graph(fuzzy_outputs, output_values):
    """ Media ponderada"""
    output_values = tf.constant(output_values, shape=[len(output_values)], dtype=tf.float32)
    num = tf.reduce_sum(tf.multiply(fuzzy_outputs, output_values), axis=1)

    return num[:, None]


def fuzzy_controller(i_vars, o_var):
    # Create the input placeholder for the controller and splitted to pass
    # each column to its fuzzification_graph
    ph_input = tf.placeholder(tf.float32, name='input', shape=[None, len(i_vars)])
    xs = tf.split(ph_input, num_or_size_splits=len(i_vars), axis=1)

    # Generate each input variable fuzzification graph
    inputs = [fuzzification_graph(x, i_var) for x, i_var in zip(xs, i_vars)]

    # Generate the inference graph
    fuzzy_outputs = inference_graph(
        fuzzy_inputs=inputs,
        num_fuzzy_inputs=[i_var.fuzzy_sets for i_var in i_vars],
        num_fuzzy_outputs=len(o_var.values)
    )

    # Defuzzification graph
    defuzzification = defuzzification_graph(fuzzy_outputs, o_var.values)

    # Now we return the inputs as a placeholder with as much columns as
    # variables as a 1-column tensor.
    return ph_input, defuzzification
