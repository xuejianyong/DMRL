# DMRL
The project of DMRL focused on a developmental modular reinforcement learning architecture, which coordinates the competition and the cooperation between modules, and inspire, in a developmental approach, the generation of new modules in cases where new goals have been detected. 

## Simulation task
We investigate the performance of DMRL in a torus grid world derived from \cite{doya2002multiple}. In this hunter-prey world, the hunter agent chooses one of five possible actions: \{north (N), east (E), south (S), west (W), stay\}, and tries to catch multiple preys in a 7x7 torus grid world (as shown in Figure \ref{pursuit_problem}). Meanwhile, preys move in either one of four directions: {NE, NW, SE, SW\}, which are represented as four different kinds of targets: $ G = \{g_1, g_2, g_3, g_4\}$.
