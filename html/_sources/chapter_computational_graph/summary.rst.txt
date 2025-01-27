
Chapter Summary
---------------

-  The computational graph technology is introduced to machine learning
   frameworks in order to achieve a trade-off between programming
   flexibility and computational efficiency.

-  A computational graph contains tensors (as units of data) and
   operators (as units of operations).

-  A computational graph represents the computational logic and status
   of a machine learning model and offers opportunities for
   optimizations.

-  A computational graph is a directed acyclic graph. Operators in the
   graph are directly or indirectly dependent on or independent of each
   other, without circular dependencies.

-  Control flows, represented by conditional control and loop control,
   determines how data flows in a computational graph.

-  Computational graphs come in two types: static and dynamic.

-  Static graphs support easy model deployment, offering high
   computational efficiency and low memory footprint at the expense of
   debugging performance.

-  Dynamic graphs provide computational results on the fly, which
   increases programming flexibility and makes debugging easy for model
   optimization and iterative algorithm improvement.

-  We can appropriately schedule the execution of operators based on
   their dependencies reflected in computational graphs.

-  For operators that run independently, we can consider concurrent
   scheduling to achieve parallel computing. For operators with
   computational dependencies, schedule them to run in serial.

-  Specific training tasks of a computational graph can run
   synchronously or asynchronously. The asynchronous mechanism
   effectively improves the hardware efficiency and shortens the
   training time.

Further Reading
---------------

-  Computational graph technology is fundamentally important to major
   machine learning frameworks. For the design details of major machine
   learning frameworks, see `TensorFlow: Large-Scale Machine Learning on
   Heterogeneous Distributed
   Systems <https://arxiv.org/abs/1603.04467>`__\ 、 `Pytorch: An
   Imperative Style, High-Performance Deep Learning
   Library <https://arxiv.org/abs/1912.01703>`__.

-  Out-of-graph control flows are created using the frontend language,
   which are easy to grasp for most programmers. However, implementing
   control flows using the in-graph approach could be challenging. For
   more on this topic, see `Implementation of Control Flow in
   TensorFlow <http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf>`__.

-  For the design and practices of dynamic and static graphs, see
   `TensorFlow Eager: A Multi-Stage, Python-Embedded DSL for Machine
   Learning <https://arxiv.org/pdf/1903.01855.pdf>`__, `Eager Execution:
   An imperative, define-by-run interface to
   TensorFlow <https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html>`__,
   `Introduction to graphs and
   tf.function <https://www.tensorflow.org/guide/intro_to_graphs>`__.
