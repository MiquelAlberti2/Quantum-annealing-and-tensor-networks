import copy
import enum
from typing import Dict, List, Tuple, Union

import numpy as np

### Utils ###


def _check_constant(c):
    if isinstance(c, (float, int, complex)):
        return HamiltonianConstantVar(c)
    return c


def _multiply_operators(op1, op2):
    if op1 not in [Operation.ADD, Operation.SUB] or op2 not in [Operation.ADD, Operation.SUB]:
        raise ValueError("only subtraction and additions are supported")
    if op1 is Operation.SUB and op2 is Operation.SUB:
        return Operation.ADD
    elif op1 is Operation.SUB or op2 is Operation.SUB:
        return Operation.SUB
    return Operation.ADD


def apply_operation(v1, op, v2):
    if op is Operation.ADD:
        return v1 + v2
    if op is Operation.SUB:
        return v1 - v2
    if op is Operation.MUL:
        return v1 * v2
    if op is Operation.DIV:
        return v1 / v2


def apply_operation_on_constants(const_list: List[Tuple["Operation", int, "HamiltonianConstantVar"]], operation):
    total_const = None  # 0 if (operation is Operation.SUB or operation is Operation.ADD) else 1
    min_i = 10000

    for op, i, con in const_list:
        v = con.value
        # if op in [Operation.ADD, Operation.SUB]:
        #     v = apply_operation(0, op, con.value)
        if total_const is None:
            total_const = 0 if (op is Operation.SUB or op is Operation.ADD) else 1
        total_const = apply_operation(total_const, op, v)
        if i < min_i:
            min_i = i
    if operation is Operation.SUB and min_i > 0:
        total_const *= -1
    return min_i, HamiltonianConstantVar(total_const)


def compare_vars(v1: "HamiltonianVariable", v2: "HamiltonianVariable"):
    if v1.label != v2.label:
        return False
    if v1.__class__ is not v2.__class__:
        return False
    if v1.bounds != v2.bounds:
        return False
    if isinstance(v1, ContinuousVar):
        if v1.encoding is not v2.encoding:
            return False
    if isinstance(v1, HamiltonianConstantVar):
        if v1.value is not v2.value:
            return False

    return True


def _multiply_pauli(op1: "PauliOperator", op2: "PauliOperator"):
    if op1.qubit_id != op2.qubit_id:
        raise ValueError("Operators must act on the same qubit")
    if op1.name == "X":
        if op2.name == "X":
            return 1, I(0)
        if op2.name == "Y":
            return 1j, Z(op1.qubit_id)
        if op2.name == "Z":
            return -1j, Y(op1.qubit_id)
        if op2.name == "I":
            return 1, op1
    if op1.name == "Y":
        if op2.name == "X":
            return -1j, Z(op1.qubit_id)
        if op2.name == "Y":
            return 1, I(0)
        if op2.name == "Z":
            return 1j, X(op1.qubit_id)
        if op2.name == "I":
            return 1, op1
    if op1.name == "Z":
        if op2.name == "X":
            return 1j, Y(op1.qubit_id)
        if op2.name == "Y":
            return -1j, X(op1.qubit_id)
        if op2.name == "Z":
            return 1, I(0)
        if op2.name == "I":
            return 1, op1
    if op1.name == "I":
        if op2.name == "X":
            return 1, X(op1.qubit_id)
        if op2.name == "Y":
            return 1, Y(op1.qubit_id)
        if op2.name == "Z":
            return 1, Z(op1.qubit_id)
        if op2.name == "I":
            return 1, I(0)
    raise NotImplementedError(f"Operation between operator {op1.name} and operator {op2.name} is not supported.")


def _multiply_pauli_and_term(out, other):
    if not (isinstance(out, Hamiltonian) and isinstance(other, PauliOperator)) and not (
        isinstance(other, Hamiltonian) and isinstance(out, PauliOperator)
    ):
        raise ValueError("You need to provide at least one Term and one Pauli Operator")

    if (isinstance(out, Hamiltonian) and out.operation is not Operation.MUL) or (
        isinstance(other, Hamiltonian) and other.operation is not Operation.MUL
    ):
        raise ValueError("only multiplication operators are supported at the moment.")

    if isinstance(out, Hamiltonian):
        ## Hamiltonian Term x Pauli operator
        ham = copy.copy(out)
        term = copy.copy(other)
        ham.elements.append(term)

    else:
        ## Pauli operator x Hamiltonian Term
        ham = copy.copy(other)
        term = copy.copy(out)
        ham.elements.insert(0, term)

    elements_dict = {}
    pop_list = []
    for i, e in enumerate(ham.elements):
        if isinstance(e, PauliOperator):
            if e.qubit_id not in elements_dict.keys():
                elements_dict[e.qubit_id] = []
            elements_dict[e.qubit_id].append(e)
            pop_list.append(i)

    pop_list = sorted(pop_list, reverse=True)
    for pop in pop_list:
        ham.elements.pop(pop)

    coeff = 1
    for k, v in elements_dict.items():
        if len(v) > 1:
            op = v[0]
            for i in range(1, len(v)):
                aux_coeff, op = _multiply_pauli(op, v[i])
                coeff *= aux_coeff

        else:
            op = v[0]

        ham.elements.append(op)

    ham *= coeff

    if len(ham.elements) == 1:
        return ham.elements[0]
    return ham


class Side(enum.Enum):
    RIGHT = "right"
    LEFT = "left"


### Operations ###


class Operation(enum.Enum):
    MUL = "*"
    ADD = "+"
    DIV = "/"
    SUB = "-"


class ComparisonOperators(enum.Enum):
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="


### Variables ###


class HamiltonianVariable:
    def __init__(self, label: str):
        self._label = label

    @property
    def label(self):
        return self._label

    def replace_variables(self, var_dict):
        if self.label in var_dict.keys():
            self.bounds = var_dict[self.label].bounds

    def __copy__(self):
        return HamiltonianVariable(label=self.label)

    def __repr__(self):
        return f"{self._label}"

    def __str__(self):
        return f"{self._label}"

    def __add__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
        if isinstance(other, Hamiltonian):
            return other + self

        out = Hamiltonian(
            elements=[self, other],
            operation=Operation.ADD,
        )
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __mul__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                return 0
            return Hamiltonian(elements=[other, self], operation=Operation.MUL)
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(elements=[self], operation=Operation.MUL) * other
        return Hamiltonian(elements=[self, other], operation=Operation.MUL)

    def __truediv__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other

        return Hamiltonian(elements=[self, _check_constant(other)], operation=Operation.DIV)

    def __sub__(self, other):
        other = _check_constant(other)
        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self

        out = Hamiltonian(elements=[self, -1 * other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __iadd__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
        if isinstance(other, Hamiltonian):
            return other + self

        out = Hamiltonian(
            elements=[self, other],
            operation=Operation.ADD,
        )
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __imul__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                return 0
            return Hamiltonian(elements=[other, self], operation=Operation.MUL)
        if isinstance(other, Hamiltonian):
            return Hamiltonian(elements=[self], operation=Operation.MUL) * other
        return Hamiltonian(elements=[self, other], operation=Operation.MUL)

    def __itruediv__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other

        return Hamiltonian(elements=[self, _check_constant(other)], operation=Operation.DIV)

    def __isub__(self, other):
        other = _check_constant(other)
        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self

        out = Hamiltonian(elements=[self, -1 * other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __radd__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
        if isinstance(other, Hamiltonian):
            return other + self

        out = Hamiltonian(
            elements=[other, self],
            operation=Operation.ADD,
        )
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __rmul__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                return 0
        if isinstance(other, Hamiltonian):
            return other * self
        return Hamiltonian(elements=[other, self], operation=Operation.MUL)

    def __rtruediv__(self, other):
        return Hamiltonian(elements=[_check_constant(other), self], operation=Operation.DIV)

    def __rfloordiv__(self, other):
        return Hamiltonian(elements=[_check_constant(other), self], operation=Operation.DIV)

    def __rsub__(self, other):
        other = _check_constant(other)
        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self

        out = Hamiltonian(elements=[other, -1 * self], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out


class HamiltonianConstantVar(HamiltonianVariable):
    def __init__(self, value):
        super().__init__(f"{value}")
        self._value = value

    @property
    def value(self):
        return self._value

    def __copy__(self):
        return HamiltonianConstantVar(value=self.value)

    def __repr__(self):
        return (
            f"{np.round(np.real_if_close(self.value), 5)}"
            if np.isreal(self.value) and np.real_if_close(self.value) >= 0
            else f"({np.round(np.real_if_close(self.value), 5)})"
        )

    def __str__(self):
        return (
            f"{np.round(np.real_if_close(self.value), 5)}"
            if np.isreal(self.value) and np.real_if_close(self.value) >= 0
            else f"({np.round(np.real_if_close(self.value), 5)})"
        )

    def __add__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
            val = self.value + other.value
            return HamiltonianConstantVar(val)

        return super().__add__(other)

    def __mul__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                return 0
            val = self.value * other.value
            return HamiltonianConstantVar(val)
        if self.value == 1:
            return other
        if self.value == 0:
            return 0
        return super().__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, HamiltonianConstantVar):
            val = self.value / other.value
            return HamiltonianConstantVar(val)
        return super().__truediv__(other)

    def __sub__(self, other):
        other = _check_constant(other)
        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
            val = self.value - other.value
            return HamiltonianConstantVar(val)
        return super().__sub__(other)

    def __iadd__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
            val = self.value + other.value
            return HamiltonianConstantVar(val)
        if self.value == 0:
            return other
        return super().__iadd__(other)

    def __imul__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                return 0
            val = self.value * other.value
            return HamiltonianConstantVar(val)
        if self.value == 1:
            return other
        if self.value == 0:
            return 0
        return super().__imul__(other)

    def __itruediv__(self, other):
        if isinstance(other, HamiltonianConstantVar):
            val = self.value / other.value
            return HamiltonianConstantVar(val)
        return super().__itruediv__(other)

    def __isub__(self, other):
        other = _check_constant(other)
        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
            val = self.value - other.value
            return HamiltonianConstantVar(val)
        return super().__isub__(other)

    def __radd__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
            val = other.value + self.value
            return HamiltonianConstantVar(val)

        return super().__radd__(other)

    def __rmul__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                return 0
            val = other.value * self.value
            return HamiltonianConstantVar(val)
        if self.value == 1:
            return other
        if self.value == 0:
            return 0
        return super().__rmul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, HamiltonianConstantVar):
            val = other.value / self.value
            return HamiltonianConstantVar(val)

        return super().__rtruediv__(other)

    def __rfloordiv__(self, other):
        if isinstance(other, HamiltonianConstantVar):
            val = other.value / self.value
            return HamiltonianConstantVar(val)
        return super().__rfloordiv__(other)

    def __rsub__(self, other):
        other = _check_constant(other)
        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return self
            val = other.value - self.value
            return HamiltonianConstantVar(val)
        return super().__rsub__(other)


### Pauli ###


class PauliOperator(HamiltonianVariable):
    """Abstract Representation of a generic Pauli operator

    Args:
        - qubit_id (int): the qubit that the operator will be acting on
        - name (str): the name of the Pauli operator
        - matrix : the matrix representation of the Pauli operator

    Attributes:
        - qubit_id (int): the qubit that the operator will be acting on
        - name (str): the name of the Pauli operator
        - matrix : the matrix representation of the Pauli operator

    Methods:
        - parse: yields the operator in the following format
                    (1, [<operator>])
    """

    def __init__(self, qubit_id: int, name: str, matrix=None) -> None:
        super().__init__(label=name)
        self.qubit_id = qubit_id
        self.__matrix = matrix
        self._name = name

    @property
    def name(self) -> str:
        """The name of the Pauli operator.

        Returns:
            str: the name of the pauli operator.
        """
        return self._name

    @property
    def matrix(self):
        """The matrix of the Pauli operator.

        Returns:

        """
        return self.__matrix

    def parse(self):
        yield 1, [self]

    def __copy__(self):
        return PauliOperator(name=self.name, qubit_id=self.qubit_id, matrix=self.matrix)

    def __repr__(self):
        return f"{self._label}({self.qubit_id})"

    def __str__(self):
        return f"{self._label}({self.qubit_id})"

    def __mul__(self, other):
        if isinstance(other, PauliOperator) and self.qubit_id == other.qubit_id:
            coeff, op = _multiply_pauli(self, other)
            return coeff * op
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, PauliOperator) and self.qubit_id == other.qubit_id:
            coeff, op = _multiply_pauli(self, other)
            return coeff * op
        return super().__rmul__(other)

    def __imul__(self, other):
        if isinstance(other, PauliOperator) and self.qubit_id == other.qubit_id:
            coeff, op = _multiply_pauli(self, other)
            return coeff * op
        return super().__mul__(other)


class Z(PauliOperator):
    """The Pauli Z operator"""

    def __init__(self, qubit_id: int) -> None:
        """constructs a new Pauli Z operator

        Args:
            qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[1, 0], [0, -1]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return Z(qubit_id=self.qubit_id)


class X(PauliOperator):
    """The Pauli X operator"""

    def __init__(self, qubit_id: int) -> None:
        """Constructs a new Pauli X operator

        Args:
            - qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[0, 1], [1, 0]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return X(qubit_id=self.qubit_id)


class Y(PauliOperator):
    """The Pauli Y operator"""

    def __init__(self, qubit_id: int) -> None:
        """Constructs a new Pauli Y operator

        Args:
            qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[0, 1j], [1j, 0]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return Y(qubit_id=self.qubit_id)


class I(PauliOperator):
    """The Identity operator"""

    def __init__(self, qubit_id: int) -> None:
        """Create a new Identity operator

        Args:
            qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[1, 0], [0, 1]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return I(qubit_id=self.qubit_id)


### Terms ###


class Hamiltonian:
    def __init__(self, elements: List[Union[HamiltonianVariable, "Hamiltonian"]], operation: Operation) -> None:
        self._elements = []
        for e in elements:
            self.elements.append(copy.copy(_check_constant(e)))
        if not isinstance(operation, Operation):
            raise ValueError(f"parameter operation expected type {Operation} but received type {operation.__class__}")
        self._op = operation

    @property
    def elements(self):
        return self._elements

    @property
    def operation(self):
        return self._op

    def variables(self):
        for e in self.elements:
            if isinstance(e, HamiltonianConstantVar):
                pass
            elif isinstance(e, HamiltonianVariable):
                yield e
            else:
                yield from e.variables()

    def to_list(self):
        output = []
        if len(self.elements) > 1:
            for i in range(len(self.elements) - 1):
                if isinstance(self.elements[i], Hamiltonian):
                    output.append(self.elements[i].to_list())
                else:
                    output.append(self.elements[i])
                output.append(self.operation)

            if isinstance(self.elements[-1], Hamiltonian):
                output.append(self.elements[-1].to_list())
            else:
                output.append(self.elements[-1])
        else:
            output = [self.elements[0]] if len(self.elements) > 0 else []
        return output

    def parse(self):
        if self.operation is Operation.MUL:
            coeff = 1
            elements = []
            for e in self.elements:
                if isinstance(e, HamiltonianConstantVar):
                    if coeff == 1:
                        coeff = e.value
                    else:
                        coeff *= e.value
                elif isinstance(e, PauliOperator):
                    elements.append(e)
                else:
                    raise NotImplementedError("parsing terms in multiplications is not supported")
            yield coeff, elements
        elif self.operation is Operation.ADD:
            for e in self.elements:
                if isinstance(e, HamiltonianConstantVar):
                    yield e.value, []
                elif isinstance(e, PauliOperator):
                    yield 1, [e]
                else:
                    yield from e.parse()

    def replace_variables(self, var_dict):
        for i, _ in enumerate(self.elements):
            self.elements[i].replace_variables(var_dict)

    def update_variables_precision(self, var_dict: Dict):
        """
        var_dict = {'<var_name>' : {'var' : <var>, 'precision': #number}}

        Args:
            var_dict (dict): _description_
        """

        if self.operation is Operation.ADD:
            for i, e in enumerate(self.elements):
                if isinstance(e, Hamiltonian):
                    if e.operation is Operation.MUL:
                        for var in e.variables():
                            if var.label in var_dict.keys():
                                e /= var_dict[var.label]["precision"]
                                self.elements[i] = e
                elif isinstance(e, HamiltonianVariable):
                    if e.label in var_dict.keys():
                        e /= var_dict[e.label]["precision"]
                        self.elements[i] = e
        elif self.operation is Operation.MUL:
            exp = copy.copy(self)
            for var in self.variables():
                if var.label in var_dict.keys():
                    exp /= var_dict[var.label]["precision"]
                    self._elements = exp.elements

    def update_negative_variables_range(self, var_dict: Dict):
        """
        var_dict = {'<var_name>' : {'var' : <var>, 'precision': #number, 'original_bounds': (#number, #number)}}

        Args:
            var_dict (dict): _description_
        """
        out = copy.copy(self)

        if out.operation is Operation.ADD:
            for i, e in enumerate(out.elements):
                if isinstance(e, Hamiltonian):
                    if e.operation is Operation.MUL:
                        for var in e.variables():
                            if var.label in var_dict.keys():
                                out += var_dict[var.label]["precision"]
                elif isinstance(e, HamiltonianVariable):
                    if e.label in var_dict.keys():
                        out += var_dict[e.label]["precision"]
        self._elements = out.elements

    def collect_constants(self):
        constants = []
        pop_list = []
        for i, e in enumerate(self.elements):
            if isinstance(e, Hamiltonian):
                if e.operation in [Operation.ADD, Operation.SUB] and self.operation in [Operation.ADD, Operation.SUB]:
                    constants.extend(e.collect_constants())
            elif isinstance(e, HamiltonianConstantVar):
                constants.append((self.operation if i != 0 else Operation.ADD, i, e))
                pop_list.append(i)

        for i in sorted(pop_list, reverse=True):
            self.elements.pop(i)
        return constants

    def simplify_constants(self, maintain_index=False):
        constants = []
        pop_list = []
        out = Hamiltonian(elements=self.elements, operation=self.operation)
        for i, e in enumerate(out.elements):
            if isinstance(e, Hamiltonian):
                if e.operation in [Operation.ADD, Operation.SUB] and out.operation in [Operation.ADD, Operation.SUB]:
                    constants.extend(e.collect_constants())
                else:
                    out.elements[i] = out.elements[i].simplify_constants()
                    if isinstance(out.elements[i], HamiltonianConstantVar):
                        constants.append((out.operation if i != 0 else Operation.ADD, i, e))
                        pop_list.append(i)
            elif isinstance(e, HamiltonianConstantVar):
                constants.append((out.operation if i != 0 else Operation.ADD, i, e))
                pop_list.append(i)

        for i in sorted(pop_list, reverse=True):
            out.elements.pop(i)

        i, out_const = apply_operation_on_constants(constants, out.operation)
        if out_const.value == 0:
            if out.operation in [Operation.ADD, Operation.SUB]:
                if len(out.elements) == 1:
                    return out.elements[0]
            elif out.operation in [Operation.MUL]:
                return 0
        elif out_const.value == 1:
            if out.operation in [Operation.MUL]:
                return out

        if len(constants) > 0:
            out.elements.insert(i if maintain_index else 0, out_const)

        return out

    def parse_parentheses(self, parent_operation=Operation.ADD):
        parsed_list = []
        op = _multiply_operators(self.operation, parent_operation)
        for i, element in enumerate(self.elements):
            if isinstance(element, Hamiltonian) and element.operation in [Operation.ADD, Operation.SUB]:
                if i == 0:
                    parsed_list.extend(element.parse_parentheses())
                else:
                    parsed_list.extend(element.parse_parentheses(parent_operation=op))

            else:
                if i == 0:
                    parsed_list.append((Operation.ADD, element))
                else:
                    parsed_list.append((op, element))
        return parsed_list

    def unfold_parentheses(self, other):
        self_elements = []
        self_is_parentheses = False
        other_elements = []
        other_is_parentheses = False

        if self.operation in [Operation.ADD, Operation.SUB]:
            self_elements = self.parse_parentheses()
            self_is_parentheses = True
        else:
            self_elements = [self]
        if isinstance(other, Hamiltonian):
            if other.operation in [Operation.ADD, Operation.SUB]:
                other_elements = other.parse_parentheses()
                other_is_parentheses = True
            else:
                other_elements = [other]
        else:
            other_elements = [other]

        output = 0
        if self_is_parentheses and other_is_parentheses:
            # two parentheses
            for op1, el1 in self_elements:
                for op2, el2 in other_elements:
                    output = apply_operation((output), _multiply_operators(op1, op2), (el1 * el2))
        elif self_is_parentheses:
            for op1, el1 in self_elements:
                for mul in other_elements:
                    output = apply_operation((output), op1, (el1 * mul))
        elif other_is_parentheses:
            for op2, el2 in other_elements:
                for mul in self_elements:
                    output = apply_operation((output), op2, (mul * el2))

        return output

    def create_hashable_term_name(self):
        """
        Assumptions:
            1. the operation is a multiplication
        """
        if self.operation is not Operation.MUL:
            raise ValueError(f"only terms with operation = {Operation.MUL.name} are allowed to be hashed")

        coeff = 1
        var_list = []

        for t in self.elements:
            if isinstance(t, HamiltonianConstantVar):
                coeff = t.value
            elif isinstance(t, HamiltonianVariable):
                var_list.append(t)

        all_pauli = True
        for v in var_list:
            if not isinstance(v, PauliOperator):
                all_pauli = False

        if all_pauli:
            var_list = sorted(var_list, key=lambda v: v.qubit_id)
        hash_name = ""
        for i in var_list:
            hash_name += f"{i}"

        return hash_name, coeff, var_list

    def simplify_variable_coefficients(self):
        """
        Assumptions:
          1. The operation is an addition
          2. only takes into account terms that are from the form: coefficient * variables
        """
        if self.operation is None:
            return
        if self.operation not in [Operation.ADD, Operation.SUB]:
            raise ValueError("the simplification is only supported for addition")

        out = Hamiltonian(self.elements, self.operation)

        hash_list = {}
        pop_list = []
        for i, e in enumerate(out.elements):
            if isinstance(e, Hamiltonian) and e.operation is Operation.MUL:
                name, coeff, var_list = e.create_hashable_term_name()
                if name not in hash_list.keys():
                    hash_list[name] = [coeff, var_list]
                else:
                    hash_list[name][0] += coeff
                pop_list.append(i)
            elif isinstance(e, HamiltonianVariable) and not isinstance(e, HamiltonianConstantVar):
                if f"{e}" not in hash_list.keys():
                    hash_list[f"{e}"] = [1, [e]]
                else:
                    hash_list[f"{e}"][0] += 1
                pop_list.append(i)
        pop_list = sorted(pop_list, reverse=True)

        for i in pop_list:
            out.elements.pop(i)

        for k, v in hash_list.items():
            term = _check_constant(v[0])
            for var in v[1]:
                term *= var

            term = _check_constant(term)
            if isinstance(term, HamiltonianConstantVar) and term.value == 0:
                continue
            out.elements.append(term)

        if len(out.elements) == 1:
            return out.elements[0]
        elif len(out.elements) == 0:
            return 0
        return out

    def simplify_pauli(self):
        ham = Hamiltonian(self.elements, self.operation)

        elements_dict = {}
        pop_list = []
        for i, e in enumerate(ham.elements):
            if isinstance(e, PauliOperator):
                if e.qubit_id not in elements_dict.keys():
                    elements_dict[e.qubit_id] = []
                elements_dict[e.qubit_id].append(e)
                pop_list.append(i)

        pop_list = sorted(pop_list, reverse=True)
        for pop in pop_list:
            ham.elements.pop(pop)

        coeff = 1
        for k, v in elements_dict.items():
            if len(v) > 1:
                op = v[0]
                for i in range(1, len(v)):
                    aux_coeff, op = _multiply_pauli(op, v[i])
                    coeff *= aux_coeff

            else:
                op = v[0]

            ham.elements.append(op)

        ham *= coeff

        if len(ham.elements) == 1:
            return ham.elements[0]
        return ham

    def __copy__(self):
        elements = []
        for e in self.elements:
            elements.append(copy.copy(e))

        return Hamiltonian(elements=elements, operation=self.operation)

    def __repr__(self):
        output_string = ""
        if len(self.elements) > 1:
            for i in range(len(self.elements) - 1):
                if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                    self.elements[i], Hamiltonian
                ):
                    output_string += f"({self.elements[i]})"
                else:
                    output_string += f"{self.elements[i]}"
                output_string += f"{self.operation.value}"
            if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                self.elements[-1], Hamiltonian
            ):
                output_string += f"({self.elements[-1]})"
            else:
                output_string += f"{self.elements[-1]}"
        else:
            output_string = f"{self.elements[0]}" if len(self.elements) > 0 else ""
        return output_string
        # if (self.operation is Operation.MUL and (isinstance(self.lhs, Term) or isinstance(self.rhs, Term))) or (
        #     self.operation is Operation.DIV and (isinstance(self.lhs, Term) or isinstance(self.rhs, Term))
        # ):
        #     return f"({str(self.lhs)}) {self.operation.value} ({str(self.rhs)})"
        # return f"{str(self.lhs)} {self.operation.value} {str(self.rhs)}"

    def __str__(self):
        output_string = ""
        if len(self.elements) > 1:
            for i in range(len(self.elements) - 1):
                if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                    self.elements[i], Hamiltonian
                ):
                    output_string += f"({self.elements[i]})"
                else:
                    output_string += f"{self.elements[i]}"

                output_string += f"{self.operation.value}"

            if (self.operation is Operation.MUL or self.operation is Operation.DIV) and isinstance(
                self.elements[-1], Hamiltonian
            ):
                output_string += f"({self.elements[-1]})"
            else:
                output_string += f"{self.elements[-1]}"
        else:
            output_string = f"{self.elements[0]}" if len(self.elements) > 0 else ""
        return output_string

    def __add__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return out
        if out.operation is Operation.ADD:
            if isinstance(other, Hamiltonian) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                out.elements.append(other)

            out = out.simplify_constants()
            if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
                out = out.simplify_variable_coefficients()
            return out
        elif out.operation is Operation.SUB:
            out = Hamiltonian(elements=[out, other], operation=Operation.ADD)
            out = out.simplify_constants(maintain_index=True)
            if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
                out = out.simplify_variable_coefficients()
            return out
        elif isinstance(other, Hamiltonian) and other.operation is Operation.ADD:
            other.elements.insert(0, out)
            other = other.simplify_constants()
            if isinstance(other, Hamiltonian) and other.operation is Operation.ADD:
                other = other.simplify_variable_coefficients()
            return other

        out = Hamiltonian(elements=[out, other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __mul__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return out
            elif other.value == 0:
                return 0

        if out.operation is Operation.MUL:
            if isinstance(other, PauliOperator):
                out = _multiply_pauli_and_term(out, other)
                out = out.simplify_constants()
                return out
            if isinstance(other, Hamiltonian) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                if isinstance(other, Hamiltonian) and other.operation in [Operation.ADD, Operation.SUB]:
                    return out.unfold_parentheses(other)
                out.elements.append(other)
            out = out.simplify_pauli()
            out = out.simplify_constants()
            return out

        elif out.operation is Operation.DIV:
            out._elements[0] *= other
            return out

        elif out.operation in [Operation.ADD, Operation.SUB]:
            return out.unfold_parentheses(other)
        return Hamiltonian(elements=[out, other], operation=Operation.MUL)

    def __truediv__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other

        return Hamiltonian(elements=[self, _check_constant(other)], operation=Operation.DIV)

    def __sub__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return out
        return out + (-1 * other)

    def __iadd__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return out
        if out.operation is Operation.ADD:
            if isinstance(other, Hamiltonian) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                out.elements.append(other)
            out = out.simplify_constants()

            if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
                out = out.simplify_variable_coefficients()
            return out
        elif isinstance(other, Hamiltonian) and other.operation is Operation.ADD:
            other.elements.insert(0, out)
            other = other.simplify_constants()
            if isinstance(other, Hamiltonian) and other.operation is Operation.ADD:
                other = other.simplify_variable_coefficients()
            return other

        out = Hamiltonian(elements=[out, other], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __imul__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return out
            elif other.value == 0:
                return 0

        if out.operation is Operation.MUL:
            if isinstance(other, PauliOperator):
                out = _multiply_pauli_and_term(out, other)
                out = out.simplify_constants()
                return out
            if isinstance(other, Hamiltonian) and other.operation == out.operation:
                out.elements.extend(other.elements)
            else:
                if isinstance(other, Hamiltonian) and other.operation in [Operation.ADD, Operation.SUB]:
                    return out.unfold_parentheses(other)
                out.elements.append(other)
            out = out.simplify_pauli()
            out = out.simplify_constants()
            return out

        elif out.operation is Operation.DIV:
            out._elements[0] *= other
            return out
        elif out.operation in [Operation.ADD, Operation.SUB]:
            return out.unfold_parentheses(other)
        return Hamiltonian(elements=[out, other], operation=Operation.MUL)

    def __itruediv__(self, other):
        other = _check_constant(other)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return self
            elif other.value == 0:
                raise ValueError("Division by zero is not allowed")

            _, other = apply_operation_on_constants([(Operation.DIV, 0, other)], Operation.DIV)  # convert it to 1/other
            return self * other
        return Hamiltonian(elements=[self, _check_constant(other)], operation=Operation.DIV)

    def __isub__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return out
        return out + (-1 * other)

    def __radd__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return out
        if out.operation is Operation.ADD:
            if isinstance(other, Hamiltonian) and other.operation == out.operation:
                other.elements.extend(out.elements)
                return other
            else:
                t = Hamiltonian(elements=[other, *out.elements], operation=Operation.ADD)
                t = t.simplify_constants()

                if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
                    t = t.simplify_variable_coefficients()
                return t
        elif isinstance(other, Hamiltonian) and other.operation is Operation.ADD:
            other.elements.append(out)
            other.simplify_constants()
            if isinstance(other, Hamiltonian) and other.operation is Operation.ADD:
                other = other.simplify_variable_coefficients()
            return other

        out = Hamiltonian(elements=[other, out], operation=Operation.ADD)
        out = out.simplify_constants(maintain_index=True)
        if isinstance(out, Hamiltonian) and out.operation is Operation.ADD:
            out = out.simplify_variable_coefficients()
        return out

    def __rmul__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 1:
                return out
            elif other.value == 0:
                return 0
        if out.operation is Operation.MUL:
            if isinstance(other, PauliOperator):
                out = _multiply_pauli_and_term(other, out)
                out = out.simplify_constants()
                return out
            if isinstance(other, Hamiltonian) and other.operation == out.operation:
                other.elements.extend(out.elements)
                return other
            else:
                if isinstance(other, Hamiltonian) and other.operation in [Operation.ADD, Operation.SUB]:
                    return out.unfold_parentheses(other)
                t = Hamiltonian(elements=[other, *out.elements], operation=Operation.MUL)
                t = t.simplify_pauli()
                t = t.simplify_constants()
                return t

        elif out.operation in [Operation.ADD, Operation.SUB]:
            return out.unfold_parentheses(other)
        return Hamiltonian(elements=[other, out], operation=Operation.MUL)

    def __rtruediv__(self, other):
        return Hamiltonian(elements=[_check_constant(other), self], operation=Operation.DIV)

    def __rfloordiv__(self, other):
        return Hamiltonian(elements=[_check_constant(other), self], operation=Operation.DIV)

    def __rsub__(self, other):
        other = _check_constant(other)
        out = Hamiltonian(self.elements, self.operation)

        if isinstance(other, HamiltonianConstantVar):
            if other.value == 0:
                return out
        return other + (-1 * out)
