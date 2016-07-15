============================================================
Modelado de comportamiento de conductores con técnicas de IA
============================================================

:Author: Alberto Díaz Álvarez

.. role:: math(raw)
   :format: html latex
..

.. role:: raw-latex(raw)
   :format: latex
..

| 

| 

| 

  Copyright ©  

 *Modelado del comportamiento de conductores con técnicas de IA*
Tesis doctoral,
Revisores: Rev1, Rev2 y Rev3
Supervisores: Dr. Francisco Serradilla García y Dr. Felipe Jiménez
Alonso

 **Universidad Politécnica de Madrid**
Instituto Universitario de Investigación del Automóvil
Campus Sur UPM, Carretera de Valencia (A-3), km7
28031 Madrid

 

*Sí sé a quién dedicárselo. Lo que no sé es cómo.*

Introducción
============

Es un hecho que la Inteligencia Artificial [1]_ (IA o AI del inglés
*Artificial Intelligence*) como área de conocimiento ha experimentado un
creciente interés en los últimos años. Esto no siempre ha sido así, ya
que después de un nacimiento muy esperanzador, con mucho optimismo
(1956) le siguieron unas épocas de apenas avance (incluso hay una época
denominada AI Winter (explicarla mejor)). Sin embargo, en la actualidad
es muy difícil encontrar un campo que no se beneficie directamente de la
aplicación de técnicas pertenecientes o nacidas en dicho área.

Esto es debido a su característica multidisciplinar ya que, si bien se
la define como área perteneciente al campo de la Informática, es
transversal a muy diferentes campos, como pueden ser por ejemplo la
biología, neurología o la psicología entre otros. [Quizá por aquí se
podría hacer una referencia al russel y norvig sobre la separación en
los cuatro cuadrantes, aunque quizá mejor en el related work].

Dentro del área de la inteligencia artificial es común diferenciar dos
tipos de aproximaciones a la hora de hablar de cómo representar el
conocimiento:

-  **Inteligencia artificial clásica** o *simbólica*. Postula que el
   conocimiento como tal es reducible a un conjunto de símbolos junto
   con operadores para su manipulación. A este tipo de técnicas y
   soluciones, que basan su funcionamiento en modelos analíticos, se las
   denominan también *Hard Computing*, y no suelen tolerar factores como
   la imprecisión y la incertidumbre (e.g. cálculo simbólico o análisis
   numérico).

-  **Inteligencia computacional** o *subsimbólica*. En ésta el
   conocimiento se alcanza por aproximación. Los esfuerzos se centran en
   la simulación de los elementos de bajo nivel que subyacen a los
   comportamientos inteligentes (e.g. redes neuronales artificiales)
   esperando que de éstos surja la solución de forma espontánea. A este
   tipo de técnicas se las conoce también como *Soft Computing*,
   conjunto de soluciones para trabajar sobre información incompleta,
   imprecisa o con ruido.

El límite entre ambos conjuntos no está perfectamente definido, máxime
si tenemos en cuenta las diferentes terminologías existentes, las
sinergias entre distintas técnicas dentro del área y los diferentes
puntos de vista sobre éstas por parte de los autores. Sin embargo, una
de las principales diferencias de ambos paradigmas es la forma de
solucionar problemas. Mientras que en el primer caso es prácticamente
impensable una aproximación diferente a la *top-down* (i.e. se
representa la solución, se define el algoritmo y éste lleva a la
solución exacta), en el segundo los problemas se resuelven utilizando el
paradigma contrario, *bottom-up* (i.e. la solución “aprende” a
resolverse dado el problema, sin necesidad de plantear un algoritmo y
generando soluciones no necesariamente exactas pero sí lo
suficientemente buenas). Revisaremos las diferencias entre conceptos de
diferentes autores en el capítulo [ch:state-of-the-art]

Uno de los campos de aplicación es el de los sistemas de transporte
inteligentes (ITS, del inglés *Intelligent Transport Systems*). Éstos se
definen como un conjunto de aplicaciones orientadas a gestionar el
transporte en todos sus aspectos y granularidades (e.g. conducción
eficiente, diseño de automóviles, gestión del tráfico o señalización en
redes de carreteras) para hacerlos más eficientes y seguros. El interés
en tal que en el año 2010 se publicó la directiva
2010/40/UE:raw-latex:`\cite{parliament2010directive}` donde se
estableció el marco de implantación de los ITS en la Unión Europea [2]_.

En el caso concreto del comportamiento al volante, es interesante la
evaluación de los conductores para conocer su manera de actuar en
determinados escenarios, y poder extraer información de estos que nos
permiten, por ejemplo, detectar qué factores pueden afectar más o menos
a determinados indicadores (e.g. el consumo estimado para una ruta en
concreto). Sin embargo, la evaluación en distintos escenarios puede no
ser interesante debido a limitaciones existentes, como pueden ser el
tiempo, el dinero o la peligrosidad del escenario.

Los simuladores de tráfico [3]_ son una solución para muchas de estas
limitaciones, pero suelen basar su funcionamiento en conductores y
vehículos (normalmente concebidos como una única entidad) basandose en
modelos de conductor que responden a funciones más o menos complejas,
además con pocas o ningunas opciones de personalización. Esto provoca
que dichos modelos se adapten poco al modelo de un conductor en
concreto.

Esta tesis pretende entrar en el tema de la generación de modelos de
conductor para simuladores que respondan al comportamiento de
conductores en concreto usando, para ello, técnicas pertenecientes al
campo de la Inteligencia Computacional.

Concretamente pretende desarrollar un método para el análisis de la
eficiencia de los conductores realizando, para ello, un modelo del
perfil de conducción a partir de técnicas de la Inteligencia
Computacional y aplicándolo a un entorno multiagente de simulación. Así,
una vez configurado el entorno multiagente, se podrá simular el tráfico
y estudiar aspectos como el estilo de conducción o el impacto de los
sistemas de asistencia.

Motivación
----------

Los conceptos introducidos al comienzo del capítulo obedecen a una
*necesidad* (eufemismo de problema) de la sociedad en la que vivimos, y
que afecta tanto a nuestra generación como afectará a las venideras: la
eficiencia en la conducción. Dado que es imprescindible saber que existe
un problema para arreglarlo, nada mejor que introducir primero algunos
hechos conocidos:

-  En el año 2014, el número de vehículos a nivel mundial asciende a más
   de :math:`1.200` millones:raw-latex:`\cite{oica2014motrate}`, con una
   tendencia ascendente. Reducir en un pequeño porcentaje el consumo
   durante la conducción evita la emisión de toneladas de gases
   considerados nocivos para el medio ambiente y el ser humano [4]_

-  Debemos abandonar los combustibles fósiles antes de que éstos nos
   abandonen a nosotros. Aunque existen diferentes puntos de vista
   acerca de cuándo se agotarán las reservas de petróleo, lo cierto es
   que los combustibles fósiles son recursos **finitos**. Lo más
   probable es que no se llegue a agotar debido a la ley de la oferta y
   la demanda, pero hay que recordar que el petróleo se usa como base
   para la producción de muchos y muy diferentes tipos de productos,
   como por ejemplo la vaselina, el asfalto o los plásticos.

-  Independientemente del momento en el que se agoten los recursos, hay
   que hacer notar que la emisión de gases está correlacionada con el
   aumento de la temperatura del planeta. De seguir con el ritmo de
   consumo actual, se teme llegar a un punto de no retorno con
   consecuencias catastróficas para la vida en el planeta.

-  Algo más cercano, y aun así íntimamente relacionado. La conducción
   eficiente afecta directamente a factores correlacionados con el
   número de accidentes de tráfico. Un factor de sobra conocido es el de
   la velocidad, factor relacionado no sólo con el número sino con la
   gravedad de los accidentes:raw-latex:`\cite{imprialou2016re}`.

Estos hechos son solo algunos que ponen de manifiesto la necesidad de
centrarse en el problema de cómo hacer de la conducción una actividad
más eficiente y segura.

La **conducción eficiente** o *eco-driving* es definida como la
aplicación de una serie de reglas de conducción con el objetivo de
reducir el consumo de carburante (en el caso de coches de combustión) o
de electricidad (en el caso de coches eléctricos).

Ser capaces de identificar o al menos estimar qué conductores son
considerados como no eficientes es importante debido a que de esta
manera se pueden identificar los hábitos recurrentes en este tipo de
perfil y adecuar la formación para eliminar dichos hábitos. Más aún
teniendo en cuenta la relación existente entre la peligrosidad y algunas
conductas agresivas. Un ejemplo donde la identificación de perfiles no
eficientes pueden tener impacto claro económico y social es el de las
empresas cuya actividad se basa en el transporte de mercancías o de
personas.

Sin embargo, identificar la conducta de un conductor no es fácil, dado
que su comportamiento se ve condicionado por numerosos factores como el
estado de la ruta, el del tráfico o el estado físico o anímico. Además,
la ambigüedad de las situaciones dificulta todavía más la
identificación. Por ejemplo, un conductor puede ser clasificado en un
momento como agresivo o no eficiente en una situación únicamente porque
su comportamiento ha sido condicionado por las malas reacciones por
parte de los demás conductores.

El análisis de todos los posibles casos es una tarea prácticamente
imposible. Por ello, las simulaciones pueden dar una estimación de los
posibles resultados de un estudio en el mundo real. Las simulaciones con
sistemas multiagente [5]_ representan a los conductores como agentes
permitiendo la evaluación del comportamiento tanto individual como
general del sistema en base a sus individuos a través de iteraciones
discretas de tiempo. Si dichos agentes son obtenidos a partir de la
modelización de conductores a partir de sus datos reales, su
comportamiento en la simulación podría ser considerado como fuente de
datos para condiciones de tráfico y/o ruta no contempladas en el mundo
real. De esta forma, se dispondría de un marco de trabajo para la
comparación de diferentes conductores sin necesidad de exponerlos a
todos y cada uno de los posibles eventos posibles. También sería posible
evaluar sistemas de asistencia evitando los problemas de no
comparabilidad de condiciones del entorno entre pruebas.

Es decir, se pretende desarrollar un método para el análisis de la
eficiencia de los conductores, realizando para ello un modelo del perfil
de conducción a partir de técnicas de Inteligencia Artificial y
aplicándolo en un entorno multiagente de donde obtener el resto de
parámetros. Así, una vez configurado el entorno multiagente, se podrá
simular el tráfico y el comportamiento de los conductores dentro de éste
cuando su marcha está condicionada por factores como el tráfico,
semáforos, etcétera.

Demostrar que la evaluación de un modelo del conductor en entornos
simulados es equivalente a la evaluación de conductores en entornos
reales implica que es posible la comparación de dos conductores usando
un criterio objetivo, es decir, sin depender del estado del resto de
factores a la hora de realizar la prueba de campo. Dicho de otro modo,
implicaría que es posible comparar la eficiencia de dos conductores
independientemente del estado del tráfico e, incluso, sobre rutas
diferentes.

Objetivos
---------

El objetivo de esta tesis doctoral es la de demostrar la
hipótesis [hyp:hypothesis-1], quedando dicha demostración dentro de los
límites impuestos por los supuestos y erstricciones indicados más
adelante.

[H[hyp:hypothesis-1]] [hyp:hypothesis-1] La aplicación de técnicas
pertenecientes al campo de la Inteligencia Computacional con datos
extraídos de un entorno de micro-simulación permitirá modelar, de manera
fiel a la realidad, el comportamiento de los conductores pertenecientes
a los grupos más representativos.

Por tanto, el objetivo de la tesis es el de simular el comportamiento de
conductores en entornos de micro-simulación a partir de su
comportamiento en entornos reales usando técnicas de Inteligencia
Computacional. Para ello se consideran los siguientes objetivos
específicos:

-  Estudiar y aplicar técnicas de la Inteligencia Computacional (e.g.
   sub-simbólica) sobre el área de la conducción.

-  Implementar métodos de generación de modelos personalizados a partir
   de datos de conductores.

-  Aplicar modelos de conductores a entornos de simulación multiagente.

-  Validar los modelos de conductor contra conductores reales.

-  Estudiar la efectividad de sistemas de asistencia encaminados a
   mejorar la eficiencia y analizar el comportamiento de conductor.

Supuestos
~~~~~~~~~

-  Se supone que el comportamiento de un conductor es el comportamiento
   en línea y el comportamiento de cambio de carril [6]_.

-  Los datos de los que extraer el comportamiento se corresponderán con
   lecturas realizadas durante el día, con buena visibilidad y sin
   lluvia.

Restricciones
~~~~~~~~~~~~~

-  La resolución máxima del modelo creado es de 1Hz.

-  En el caso de los modelos que hacen uso de redes neuronales
   artificiales, no se pueden exlpicar las razones del comportamiento
   inferido.

Resultados
----------

Estructura de la tesis
----------------------

La tesis está estructurada de la siguiente manera:

-  ****. Revisión del estado de la cuestión donde se explica en qué
   punto se encuentra la literatura de los temas en los que se apoya la
   presente tesis.

Estado de la cuestión
=====================

Simuladores, micro y macro Comparativa de simuladores y por qué se ha
elegido SUMO Sistemas multiagente Qué es la inteligencia artificial.
Diferencias entre inteligencia artificial clásica e inteligencia
computacional. Diferentes puntos de vista (soft computing, machine
learning, ...) Técnicas de la inteligencia computacional usadas en esta
tesis (redes neuronales artificiales(perceptrón multicapa, recurrentes y
lstm), lógica difusa y computación evolutiva)

Aplicar técnicas de la inteligencia computacional (o de la rama
subsimbólica de la IA o del softcomputing) sobre el área de la
conducción. – ¿Qué técnicas se usan actualmente y sobre qué problemas?

Concretamente – Sobre el estudio de la efectividad de sistemas de
asistencia encaminados a mejorar la eficiecia y para el análisis del
comoprtamiento del conductos (detección de patrones de eficiencia y
agresividad de subyacen en los comportamientos de éstos). – Aquí hay dos
cosas. Por un lado Estudio de la efectividad de los sistemas de
asistencia para mejorar la eficiencia de conucción y estudio de los
sistemas de asistencia para analizar el comportamiento del conductor.

Estudio y aplicación de técnicas de la rama subsimbólica de la
inteligencia artificial sobre el área de la conducción. Concretamente
para el estudio de la efectividad de sistemas de asistencia encaminados
a mejorar la eficiencia y para el análisis de comportamiento de
conductor (detección de los modelos y patrones de eficiencia y
agresividad que subyacen en los comportamientos de los conductores).

El núcleo de la Tesis consiste en el estudio y la aplicación de técnicas
de la rama subsimbólica de la Inteligencia Artificial sobre el área de
la conducción, concretamente para el estudio de la efectividad de
sistemas de asistencia encaminados a mejorar la eficiencia y para el
análisis del comportamiento de conductor (detección de los modelos y
patrones de eficiencia y agresividad que subyacen en los comportamientos
de los conductores).

 

**Acerca del código fuente**

La presente tesis lleva consigo numerosas horas de programación, lo que
implica varios miles de líneas de código. Sin embargo, esta nota
existiria aún con sólo unas pocas decenas. Se ha decidido no proveer de
forma impresa el código fuente y en su lugar distribuirlo en formato
electrónico, una forma más manejable para su consulta y a la vez
respetuosa con el medio ambiente. No obstante sí es posible que existan
pequeños fragmentos de código para apoyar explicaciones. En caso de
necesitar los fuentes y no ser capaces de conseguirlos, se puede
contactar directamente conmigo, el autor, en
`. <mailto:alberto.diaz@upm.es>`__

**Cómo citar esta tesis**

Si deseas citar esta tesis, lo primero gracias. Me alegro de que te
sirva para tu investigación. Si lo deseas, incluye el siguiente código
en bibtex:

**TODO A ver cómo coño meto en el paquete listings caracteres
acentuados...**

.. [1]
   A pesar de existir casi tantas definiciones como autores, podemos
   definir la inteligencia artificial como el *área que estudia el
   comportamiento inteligente exhibido en máquinas*.

.. [2]
   En esta directiva, los ITS se definen como *aplicaciones avanzadas
   que, sin incluir la inteligencia como tal, proporcionan servicios
   innovadores en relación con los diferentes modos de transporte y la
   gestión del tráfico y permiten a los distintos usuarios estar mejor
   informados y hacer un uso más seguro, más coordinado y «más
   inteligente» de las redes de transporte.*

.. [3]
   Concretamente los *micro-simuladores*, donde los conductores y/o
   vehículos son modelados como agentes independientes. Existe otra
   categoría dentro de la simulación de tráfico, denominada
   *macro-simuladores*, que conciben el tráfico como un fluido. Aún así,
   veremos este concepto más en detalle en el capítulo .

.. [4]
   Uno puede argumentar que el parque automovilístico se recicla con
   nuevos vehículos eléctricos categorizados “de consumo 0”. La triste
   realidad es que estos vehículos consumen la electricidad generada
   actualmente de una mayoría de centrales de combustibles fósiles y
   nucleares. Además, mientras que en países desarrollados el
   crecimiento ha sido en torno al 4-7%, en países subdesarrollados,
   donde no existe aun infraestructura para la recarga de vehículos
   eléctricos, dicho crecimiento ha superado el 120%.

.. [5]
   Los sistemas multiagente (SMA o MAS del inglés *Multi-Agent Systems*)
   son aquellos sistemas compuestos por diversos elementos denominados
   agentes, los cuales cooperan sobre un entorno para, normalmente,
   llegar a una solución.

.. [6]
   Son conocidos en la literatura como *car-following* y *lane-changing*
   respectivamente. Entraremos en detalle sobre ambos conceptos en el
   capítulo
