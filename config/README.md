This folder contains the global configuration controlling how the workflows are run.

`workflow_config.yaml` is a configuration file that determine the correspodance between the experiment type in the `parameters.yaml` of a recording and which workflow `runSnake` should run. If you are working on a new experimental type, please create a new workflow file in the top `workflow` folder and and specify its relation with the experimental type in the `snakefiles` field.
