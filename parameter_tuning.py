from ax import optimize
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render
import numpy as np
import train

best_parameters, best_values, experiment, _ = optimize(
    parameters=[
        {"name": "weight_decay", "type": "range", "bounds": [0.001, 2.0]},
        {"name": "lr", "type": "range", "bounds": [0.0001, 0.1]},
        {"name": "c1c2", "type": "choice", "values": [i for i in range(5, 15, 1)]},
        {"name": "c2c3", "type": "choice", "values": [i for i in range(10, 25, 2)]},
        {"name": "c3c4", "type": "choice", "values": [i for i in range(20, 35, 2)]},
        {"name": "c4c5", "type": "choice", "values": [i for i in range(30, 45, 2)]},
        {"name": "c5c6", "type": "choice", "values": [i for i in range(40, 55, 2)]},
        {"name": "c6c7", "type": "choice", "values": [i for i in range(50, 70, 2)]},
        {"name": "c7l1", "type": "choice", "values": [i for i in range(60, 90, 2)]},
                ],
    #parameter_constraints=["c1c2 <= c2c3", "c2c3 <= c3c4", "c3c4 <= c4c5", "c4c5 <= c5c6", "c5c6 <= c6c7", "c6c7 <= c7l1"],
    evaluation_function=train.train_multiple, minimize=True,
    total_trials=50)
print(best_parameters)
print("Best values: ", best_values)

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])
best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Dev loss, %",
)
render(best_objective_plot)