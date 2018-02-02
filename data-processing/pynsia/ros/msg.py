def headers(message):
    """ Extract the headers of a given message.

    :param message: The message from which extract the headers.
    :return: A list with the headers in the order they appear. If there are no
        headers, the list will be empty.
    """
    # Separate all the lines
    lines = [s.strip() for s in str(message).split('\n')]

    # Store "key" from all the lines in the form key: value. Ignore the rest
    headers = []
    for line in lines:
        if ':' in line:
            header, _ = line.split(':')
            headers.append(header.strip())

    return headers


def headers_and_values(message, headers=None):
    """ Generates a dictionary where the keys are the headers.

    :param message: The message from where to extract the values.
    :param h: If set, it'll only extract those messages. If not set, all the
        headers will be extracted.
    :return: A dictionary with the headers and their associated values. If any
        of the headers are not present, its value will be None.
    """

    def global_headers():
        global headers
        headers()

    headers_and_vals = {header: None for header in headers or global_headers()}
    # Extract the lines (which potentially have key: value elements)
    lines = [s.strip() for s in str(message).split('\n')]
    # Strip the strings in the form key: value and ignore the rest
    for line in lines:
        if ':' in line:
            key, value = [x.strip() for x in line.split(':')]
            # If is a header that we want and has value, get it
            if key in headers_and_vals and value:
                headers_and_vals[key] = value

    return headers_and_vals
