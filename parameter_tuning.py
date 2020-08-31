from ax import optimize
import train

best_parameters, best_values, _, _ = optimize(
    parameters=[
        {"name": "weight_decay", "type": "range", "bounds": [0.001, 1.0]},
        {"name": "lr", "type": "range", "bounds": [0.0001, 0.01]},
        {"name": "c1c2", "type": "choice", "values": [i for i in range(5, 11, 3)]},
        {"name": "c2c3", "type": "choice", "values": [i for i in range(20, 31, 3)]},
        {"name": "c3c4", "type": "choice", "values": [i for i in range(30, 41, 3)]},
        {"name": "c4c5", "type": "choice", "values": [i for i in range(40, 61, 3)]},
        {"name": "c5c6", "type": "choice", "values": [i for i in range(60, 81, 3)]},
        {"name": "c6l1", "type": "choice", "values": [i for i in range(80, 150, 3)]}
                ],
    evaluation_function=train.train_model, minimize=True)
print(best_parameters)
print("Best values: ", best_values)