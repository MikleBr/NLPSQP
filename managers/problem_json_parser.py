import json
import numpy as np
from typing import List
from sympy import lambdify, parse_expr
from models.optimization_task import OptimizationTask
from models.optimization_config import OptimizationConfig
from models.constraint import Constraint, ConstraintType
from models.target_function import TargetFunction
from models.design_variable import DesignVariable


class ProblemJSONParser:
    def parseVariables(self, variables):
        """
        Парсит список переменных из JSON и инициализирует объекты класса DesignVariable.
        
        :param variables: Список переменных из JSON.
        :return: Массив объектов DesignVariable.
        """
        variable_list = []
        for var in variables:
            identifier = var.get("identifier")
            name = var.get("name")
            defaultValue = var.get("defaultValue")
            lowerBound = var.get("lowerBound", -np.inf)
            upperBound = var.get("upperBound", np.inf)

            if not identifier or not name or defaultValue is None:
                raise ValueError(f"Неполные данные для переменной: {var}")

            # Создаем объект DesignVariable
            design_var = DesignVariable(
                name=identifier,  # Используем identifier как имя переменной
                value=defaultValue,
                lower=lowerBound,
                upper=upperBound
            )
            variable_list.append(design_var)

        return variable_list

    def parseFunctionFromString(self, function_str, allowed_variables):
        """
        Парсит математическую функцию из строки и проверяет переменные.

        :param function_str: Строка, представляющая математическое выражение.
        :param allowed_variables: Список разрешенных переменных.
        :return: Лямбда-функция, принимающая словарь параметров.
        """
        try:
            # Парсим выражение
            expr = parse_expr(function_str)
            variables_in_expr = {str(var) for var in expr.free_symbols}

            # Проверяем, что все переменные в выражении разрешены
            if not variables_in_expr.issubset(allowed_variables):
                raise ValueError(
                    f"Выражение содержит недопустимые переменные: "
                    f"{variables_in_expr - allowed_variables}"
                )

            # Создаем исполняемую функцию
            func = lambdify(list(expr.free_symbols), expr, modules="numpy")
            return lambda params: func(**{str(var): params[str(var)] for var in expr.free_symbols})

        except Exception as e:
            print(f"Ошибка при парсинге функции: {e}")
            raise

    def createTargetFunction(self, data, allowed_variables) -> TargetFunction:
        target_function = data.get("target_function")
        if not target_function:
            raise ValueError("В файле отсутствует ключ 'target_function'")

        target_expression = target_function.get("expression")
        if not target_expression:
            raise ValueError("В 'target_function' отсутствует ключ 'expression'")

        parsed_target_function = self.parseFunctionFromString(
            target_expression, allowed_variables)

        target_name = target_function.get("name")

        return TargetFunction(parsed_target_function, target_name)

    def createConstraint(self, constraintJson, allowed_variables) -> Constraint:
        constraint_expression = constraintJson.get("expression")

        if not constraint_expression:
            raise ValueError(
                "В одном из ограничений отсутствует ключ 'expression'")

        parsed_constraint = self.parseFunctionFromString(
            constraint_expression, allowed_variables)

        constraint_name = constraintJson.get("name")

        return Constraint(parsed_constraint, ConstraintType.INEQ, constraint_name)

    def createProblem(self, filePath: str) -> OptimizationTask:
        try:
            with open(filePath, 'r') as file:
                data = json.load(file)

            variables = data.get("variables")
            if not variables:
                raise ValueError("В файле отсутствует ключ 'variables'")
            variable_list = self.parseVariables(variables)
            # Извлекаем имена переменных
            allowed_variables = {var.name for var in variable_list}

            # Шаг 3: Парсинг целевой функции
            target_function = self.createTargetFunction(data, allowed_variables)

            # Шаг 4: Парсинг ограничений
            constraints = data.get("constraints", [])
            parsed_constraints: List[Constraint] = []
            for constraint in constraints:
                constraintInstance = self.createConstraint(
                    constraint, allowed_variables)
                parsed_constraints.append(constraintInstance)

            return OptimizationTask(
                variables=variable_list,
                target=target_function,
                constraints=parsed_constraints,
                config=OptimizationConfig()
            )

        except FileNotFoundError:
            print(f"Ошибка: Файл '{self.filePath}' не найден.")
            raise
        except json.JSONDecodeError:
            print(f"Ошибка: Неверный формат JSON в файле '{self.filePath}'.")
            raise
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            raise
