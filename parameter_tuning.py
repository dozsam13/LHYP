from ax import optimize
import train

best_parameters, best_values, _, _ = optimize(
    parameters=[{"name": "weight_decay", "type": "range", "bounds": [0.001, 1.0]}],
    evaluation_function=train.train_model, minimize=False)
print(best_parameters)
