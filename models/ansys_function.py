import json
import hashlib
import os
import re
from managers.ansys import run
from models.target_function import TargetFunction

def unwrap_value(val):
    """
    Универсально преобразует autograd.ArrayBox, numpy.float, строки и прочее к float.
    """
    import numbers

    try:

        if isinstance(val, numbers.Number):
            return float(val)

        if hasattr(val, '_value'):
            if hasattr(val._value, '_value'):
                return float(val._value._value)
            return float(val._value)

        if isinstance(val, str):
            import re
            match = re.search(r"value\s+([+-]?[0-9.eE+-]+)", val)
            if match:
                return float(match.group(1))
            else:
                raise TypeError(f"Не удалось распарсить строку: {val}")


        return float(val)
    except Exception as e:
        raise TypeError(f"Невозможно привести значение '{val}' к float: {e}")


class AnsysFunction(TargetFunction):
    def __init__(
        self,
        macro_template_path: str,
        ansys_path: str,
        workdir: str,
        output_filename: str,
        result_parser,
        name="AnsysTargetFunction",
        use_cache=True
    ):
        self.template_path = macro_template_path
        self.ansys_path = ansys_path
        self.workdir = workdir
        self.output_filename = output_filename
        self.result_parser = result_parser
        self.use_cache = use_cache
        self.cache = {}
        super().__init__(self.evaluate, name)

    def _make_hash(self, variables: dict) -> str:
        sorted_items = sorted((k, float(v)) for k, v in variables.items())
        hash_str = json.dumps(sorted_items, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def evaluate(self, variables: dict) -> float:
        key = self._make_hash(variables)
        if self.use_cache and key in self.cache:
            return self.cache[key]

        with open(self.template_path, "r") as f:
            macro = f.read()

        for var, val in variables.items():
            val_float = unwrap_value(val)
            pattern = rf"(?m)^(\s*{var}\s*=\s*).*$"
            macro = re.sub(pattern, lambda m: f"{m.group(1)}{val_float}", macro)

        script_filename = "file"
        macro_path = os.path.join(self.workdir, script_filename + ".txt")
        with open(macro_path, "w") as f:
            f.write(macro)

        # 4. Запуск ANSYS
        run(script_filename, self.ansys_path, self.workdir)

        # 5. Чтение результата
        result_file = os.path.join(self.workdir, self.output_filename)
        if not os.path.exists(result_file):
            raise RuntimeError(f"Файл результата для {self.name} не найден: {result_file}")

        result = self.result_parser(result_file)

        if self.use_cache:
            self.cache[key] = result

        return result