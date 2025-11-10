import importlib
import yaml
from sklearn.model_selection import GridSearchCV

class ModelFactory:
    def __init__(self, model_config_path: str):
        with open(model_config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def _import_class(self, module_name: str, class_name: str):
        """Dynamically import any sklearn model from YAML."""
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def get_best_model(self, X, y, base_score: float):
        """Run GridSearchCV for all models and return the best one."""
        grid_params = self.config["grid_search"]["params"]
        models = self.config["model_selection"]

        best_model = None
        best_score = base_score
        best_model_name = None
        best_params = None

        for key, model_info in models.items():
            module_name = model_info["module"]
            class_name = model_info["class"]
            params = model_info["params"]
            param_grid = model_info["search_param_grid"]

            model_class = self._import_class(module_name, class_name)
            model = model_class(**params)

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                **grid_params
            )

            grid_search.fit(X, y)
            score = grid_search.best_score_

            print(f"Model: {class_name}, Score: {score:.4f}")

            if score > best_score:
                best_model = grid_search.best_estimator_
                best_score = score
                best_model_name = class_name
                best_params = grid_search.best_params_

        return {
            "best_model_name": best_model_name,
            "best_model": best_model,
            "best_score": best_score,
            "best_params": best_params
        }
