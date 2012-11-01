# -*- coding: utf-8 -*-

import inspect

import parse

class UndefinedType:
    pass

class NullType:
    pass

class NaNType:
    pass

Undefined = UndefinedType()
Null = NullType()
NaN = NaNType()

class ESException(Exception):
    pass


class ESTypeError(ESException):
    pass


class ESReferenceError(ESException):
    def __init__(self, prop_name=None):
        super().__init__(self)
        self.prop_name = prop_name

    def __str__(self):
        return self.prop_name


class Return(Exception):
    def __init__(self, value):
        self.value = value


class PropertyDescriptor:
    def __init__(self, value=None, get=None, set=None,
                 writable=None, enumerable=None, configurable=None):
        self.value = value
        self.get = get
        self.set = set
        self.writable = writable
        self.enumerable = enumerable
        self.configurable = configurable

    def is_accessor_descriptor(self):
        return self.get is not None or self.set is not None

    def is_data_descriptor(self):
        return self.value is not None or self.writable is not None

    def is_generic_descriptor(self):
        if self.is_accessor_descriptor():
            return False
        if self.is_data_descriptor():
            return False
        return True


# 9.12 The SameValue Algorithm
def same_value(x, y):
    if type(x) != type(y):
        return False
    if x is Undefined:
        return True
    if x is Null:
        return True
    if isinstance(x, float):
        return x == y
    if isinstance(x, str):
        return x == y
    if isinstance(x, bool):
        return x == y
    return x is y

# 8.12.9
def reject(throw):
    if throw:
        raise ESTypeError
    return False

class Object:
    def __init__(self, klass="Object", prototype=Null, extensible=True):
        self._props = {}
        self.klass = klass
        self.prototype = prototype
        self.extensible = extensible

    # 8.12.1 [[GetOwnProperty]](P)
    def get_own_property(self, name):
        if not name in self._props:
            return None
        desc = self._props[name]

        new_desc = PropertyDescriptor()
        if desc.is_data_descriptor():
            new_desc.value = desc.value
            new_desc.writable = desc.writable
        else:
            assert desc.is_accessor_descriptor()
            new_desc.get = desc.get
            new_desc.set = desc.set
        new_desc.enumerable = desc.enumerable
        new_desc.configurable = desc.configurable
        return new_desc
    
    # 8.12.2 [[GetProperty]](P)
    def get_property(self, name):
        prop = self.get_own_property(name)
        if prop:
            return prop
        proto = self.prototype
        if proto is Null:
            return None
        return proto.get_property(name)

    # 8.12.3 [[Get]](P)
    def get(self, name):
        desc = self.get_property(name)
        if not desc:
            return Undefined
        if desc.is_data_descriptor():
            return desc.value
        assert desc.is_accessor_descriptor()
        if not desc.get:
            return Undefined
        return desc.get.call(self)

    # 8.12.4 [[CanPut]] (P)
    def can_put(self, name):
        desc = self.get_own_property(name)
        if desc:
            if desc.is_accessor_descriptor():
                return desc.set is not None
            return desc.writable

        proto = self.prototype
        if proto is Null:
            return self.extensible

        inherited = proto.get_property(name)
        if not inherited:
            return self.extensible
        if inherited.is_accessor_descriptor():
            return inherited.set is not None
        if not self.extensible:
            return False
        return inherited.writable

    # 8.12.5 [[Put]](P, V, Throw)
    def put(self, name, value, throw):
        if not self.can_put(name):
            if throw:
                raise ESTypeError("cannot put!")
            return
        own_desc = self.get_own_property(name)
        if own_desc and own_desc.is_data_descriptor():
            value_desc = PropertyDescriptor(value=value)
            self.define_own_property(name, value_desc, throw)
            return
        desc = self.get_property(name)
        if desc and desc.is_accessor_descriptor():
            desc.set.call(self, value)
        else:
            new_desc = PropertyDescriptor(value=value, writable=True,
                                          enumerable=True, configurable=True)
            self.define_own_property(name, new_desc, throw)
        return

    # 8.12.6 [[HasProperty]](P)
    def has_property(self, name):
        return self.get_property(name) is not None
 
    _desc_field_names = ["value", "writable", "get", "set", "enumerable",
                         "configurable"]

    # 8.12.9 [[DefineOwnProperty]](P, Desc, Throw)
    def define_own_property(self, name, desc, throw):
        current = self.get_own_property(name)
        if not current:
            if not self.extensible:
                return reject()

            # Step 4.
            new_desc = PropertyDescriptor()
            if desc.enumerable is not None:
                new_desc.enumerable = desc.enumerable
            else:
                new_desc.enumerable = False
            if desc.configurable is not None:
                new_desc.configurable = desc.configurable
            else:
                new_desc.configurable = False

            if desc.is_generic_descriptor() or desc.is_data_descriptor():
                if desc.value is not None:
                    new_desc.value = desc.value
                else:
                    new_desc.value = Undefined
                if desc.writable is not None:
                    new_desc.writable = desc.writable
                else:
                    new_desc.writable = False
            else:
                new_desc.get = desc.get or Undefined
                new_desc.set = desc.set or Undefined

            self._props[name] = new_desc
            return True

        # Step 5.
        for field in Object._desc_field_names:
            if getattr(desc, field) is not None:
                break
        else:
            return True

        # Step 6.
        for field in Object._desc_field_names:
            v = getattr(desc, field)
            if v is None:
                continue
            x = getattr(current, field)
            if x is None:
                break
            if not same_value(v, x):
                break
        else:
            return True

        # Step 7.
        if current.configurable is False:
            if desc.configurable is True:
                return reject()
            if desc.enumerable is not None:
                if desc.enumerable != current.enumerable:
                    return reject()

        # Step 8.
        if desc.is_generic_descriptor():
            pass
        # Step 9.
        elif current.is_data_descriptor() != desc.is_data_descriptor():
            if current.configurable is False:
                return reject()
            if current.is_data_descriptor():
                prop = self._props[name]
                prop.value = Undefined
                prop.writable = False
                prop.get = Undefined
                prop.set = Undefined
            else:
                prop = self._props[name]
                prop.value = Undefined
                prop.writable = False
                prop.get = Undefined
                prop.set = Undefined
        # Step 10.
        elif current.is_data_descriptor() and desc.is_data_descriptor():
            if current.configurable is False:
                if current.writable is False:
                    if desc.writable is True:
                        return reject()
                    if (desc.value is not None and
                        not same_value(desc.value, current.value)):
                        return reject()
        # Step 11.
        elif current.is_accessor_descriptor() and desc.is_accessor_descriptor():
            if current.configure is False:
                if (desc.set is not None and
                    not same_value(desc.set, current.set)):
                    return reject()
                if (desc.get is not None and
                    not same_value(desc.ge, current.get)):
                    return reject()

        # Step 12.
        prop = self._props[name]
        for field in Object._desc_field_names:
            v = getattr(desc, field)
            if v is not None:
                setattr(prop, field, v)
        return True


class Function(Object):
    def __init__(self, scope, formal_params, strict):
        self.strict = strict
        super().__init__("Function", Null, True)
        self.scope = scope
        self.formal_parameters = formal_params

        desc = PropertyDescriptor(value=len(formal_params),
                                  writable=False,
                                  enumerable=False,
                                  configurable=False)
        self.define_own_property("length", desc, False)

    def _compute_this(self, this_arg):
        # 10.4.3 Entering Function Code
        if self.strict:
            return this_arg
        elif this_arg is Null or this_arg is Undefined:
            scope = self.scope
            while scope.outer:
                scope = scope.outer
            return scope
        elif not isinstance(this_arg, Object):
            return to_object(this_arg)
        else:
            return this_arg


class ScriptedFunction(Function):
    def __init__(self, scope, formal_params, code, strict):
        super().__init__(scope, formal_params, strict)
        self.code = code

    # 13.2.1 [[Call]]
    def call(self, rt, this_arg, args):
        this = self._compute_this(this_arg)

        # 10.4.3 Step 5.
        local_env = new_declarative_environment(self.scope)
        context = ExecutionContext(local_env, local_env, this)
        rt.push_context(context)
        instantiate_declaration_binding(rt, self.code, (self, args))

        # 13.2.1 Step 2.
        evaluator = Evaluator(rt, self.code)
        result = None
        try:
            self.code.accept(evaluator)
        except Return as e:
            result = e.value
        rt.pop_context()
        return result if result is not None else Undefined


class NativeFunction(Function):
    def __init__(self, scope, function, strict):
        formal_params = inspect.getargspec(function)[0]
        super().__init__(scope, formal_params, strict)
        self.code = "<native code>"
        self._function = function

    def call(self, rt, this_arg, args):
        if self.scope is None:
            self.scope = rt.global_env
        this = self._compute_this(this_arg)
        return self._function(this, *args)


class GlobalObject(Object):
    def __init__(self):
        super().__init__("Global", Null, True)


# 8.7 The Reference Specification Type
class Reference:
    def __init__(self, base, referenced_name, is_strict):
        self.base = base
        self.referenced_name = referenced_name
        self.is_strict = is_strict

    def has_primitive_base(self):
        class_ = self.base.__class__
        return class_ is float or class_ is str or class_ is bool

    def is_property_reference(self):
        return isinstance(self.base, Object) or self.has_primitive_base()

    def is_unresolvable_reference(self):
        return self.base is Undefined

# 8.7.1 GetValue(V)
def get_value(value):
    if value.__class__ is not Reference:
        return value
    if value.is_unresolvable_reference():
        raise ESReferenceError(value.referenced_name)
    base = value.base
    if value.is_property_reference():
        if not value.has_primitive_base():
            # FIXME: pass a this value
            return base.get(value.referenced_name)
        raise NotImplementedError
    return base.get_binding_value(value.referenced_name, value.is_strict)

# 8.7.2 PutValue (V, W)
def put_value(rt, V, W):
    if V.__class__ is not Reference:
        raise ESReferenceError
    if V.is_unresolvable_reference():
        if V.is_strict:
            raise ESReferenceError
        rt.global_object.put(V.referenced_name, W, False)
    else:
        V.base.set_mutable_binding(V.referenced_name, W, V.is_strict)

# 9.1 ToPrimitive
def to_primitive(input_):
    return input_

# 9.2 ToBoolean
def to_boolean(v):
    if v is False or v is True:
        return v
    if v is Undefined or v is Null:
        return False
    if v.__class__ is float:
        if v == +0 or v == -1:
            return False
        return True
    if v.__class__ is str:
        return not v
    if isinstance(v, Object):
        return True
    if v is NaN:
        return False
    raise NotImplementedError

# 9.3 ToNumber
def to_number(v):
    if v is Undefined:
        return NaN
    if v is Null:
        return (+0.0)
    return v

# 9.5 ToInt32: (Signed 32 Bit Integer)
def to_int32(v):
    return int(to_number(v))

to_uint32 = to_int32

# 9.10 CheckObjectCoercible
def check_object_coercible(value):
    if value is Undefined or value is Null:
        raise ESTypeError
    return value

# 9.11 IsCallable
def is_callable(o):
    if not isinstance(o, Object):
        return False
    return hasattr(o, "call")

# --- Lexical Environments ---------------------------------------------------

class LexicalEnvironment:
    def __init__(self, record, outer):
        self.record = record
        self.outer = outer


class EnvironmentRecord:
    pass


class Binding:
    def __init__(self, value, mutable, can_delete):
        self.value = value
        self.mutable = mutable
        self.can_delete = can_delete


# 10.2.1.1 Declarative Environment Records
class DeclarativeEnvironmentRecord(EnvironmentRecord):
    def __init__(self):
        self.bindings = {}

    # 10.2.1.1.1 HasBinding(N)
    def has_binding(self, name):
        return name in self.bindings

    # 10.2.1.1.2 CreateMutableBinding (N, D)
    def create_mutable_binding(self, name, deletion):
        assert not name in self.bindings
        self.bindings[name] = Binding(Undefined, True, deletion)

    # 10.2.1.1.3 SetMutableBinding (N,V,S)
    def set_mutable_binding(self, name, value, strict):
        binding = self.bindings.get(name)
        assert binding
        if binding.mutable:
            binding.value = value
            return
        if strict:
            raise ESTypeError

    # 10.2.1.1.4 GetBindingValue(N,S)
    def get_binding_value(self, name, strict):
        binding = self.bindings.get(name)
        assert binding
        if not binding.mutable and binding.value is Undefined:
            if not strict:
                return Undefined
            raise ESReferenceError(name)
        return binding.value

    # 10.2.1.1.5 DeleteBinding (N)
    def delete_binding(self, name):
        binding = self.bindings.get(name)
        if not binding:
            return True
        if not binding.can_delete:
            return False
        del self.bindings[name]
        return True

    # 10.2.1.1.6 ImplicitThisValue()
    def implicit_this_value(self):
        return Undefined

    # 10.2.1.1.7 CreateImmutableBinding (N)
    def create_immutable_binding(self, name):
        assert not name in self.bindings
        self.bindings[name] = Binding(Undefined, False, False)

    # 10.2.1.1.8 InitializeImmutableBinding (N,V)
    def initialize_immutable_binding(self, name, value):
        binding = self.bindings.get(name)
        assert binding
        assert not binding.mutable
        assert binding.value is Undefined
        binding.value = value


# 10.2.1.2 Object Environment Records
class ObjectEnvironmentRecord(EnvironmentRecord):
    def __init__(self, obj, privide_this=False):
        self.bindings = obj
        self.privide_this = privide_this

    # 10.2.1.2.1 HasBinding(N)
    def has_binding(self, name):
        return self.bindings.has_property(name)

    # 10.2.1.2.2 CreateMutableBinding (N, D)
    def create_mutable_binding(self, name, config_value):
        assert not self.bindings.has_property(name)
        desc = PropertyDescriptor(value=Undefined, writable=True,
                                  enumerable=True, configurable=config_value)
        self.bindings.define_own_property(name, desc, True)

    # 10.2.1.2.3 SetMutableBinding (N,V,S)
    def set_mutable_binding(self, name, value, strict):
        self.bindings.put(name, value, strict)

    # 10.2.1.2.4 GetBindingValue(N,S)
    def get_binding_value(self, name, strict):
        if not self.bindings.has_property(name):
            if not strict:
                return Undefined
            raise ESReferenceError(name)
        return self.bindings.get(name)

    # 10.2.1.2.6 ImplicitThisValue()
    def implicit_this_value(self):
        return self.bindings if self.privide_this else Undefined


#
# 10.2.2 Lexical Environment Operations
#

# 10.2.2.1 GetIdentifierReference(lex, name, strict)
def get_identifier_reference(lex, name, strict):
    if not lex:
        return Reference(Undefined, name, strict)
    env_rec = lex.record
    if env_rec.has_binding(name):
        return Reference(env_rec, name, strict)
    return get_identifier_reference(lex.outer, name, strict)

# 10.2.2.2 NewDeclarativeEnvironment (E)
def new_declarative_environment(outer_env):
    record = DeclarativeEnvironmentRecord()
    return LexicalEnvironment(record, outer_env)

# 10.2.2.3 NewObjectEnvironment (O, E)
def new_object_environment(obj, outer_env):
    record = ObjectEnvironmentRecord(obj)
    return LexicalEnvironment(record, outer_env)

# 10.3 Execution Contexts
class ExecutionContext:
    def __init__(self, lex_env, var_env, this):
        self.lex_env = lex_env
        self.var_env = var_env
        self.this = this


class FunctionDeclarationInstantiator(parse.ASTVisitor):
    def __init__(self, rt):
        self.scope = rt.context.lex_env
        self.global_env = rt.global_env
        self.global_object = rt.global_object

    def visit_function_decl(self, decl):
        # 13.2 Creating Function Objects
        f_obj = ScriptedFunction(self.scope, decl.params, decl, False)

        if not self.scope.record.has_binding(decl.name):
            self.scope.record.create_mutable_binding(decl.name, False)
        elif self.scope.record is self.global_env:
            existing_prop = self.global_object.get_property(decl.name)
            if existing_prop.configurable:
                desc = PropertyDescriptor(value=Undefined, writable=True,
                                          enumerable=True, configurable=False)
                self.global_object.define_own_property(decl.name, desc, True)
            elif (existing_prop.is_accessor_descriptor() or
                  not (existing_prop.writable is True and
                       existing_prop.enumerable is True)):
                raise ESTypeError
        self.scope.record.set_mutable_binding(decl.name, f_obj, False)


class VariableDeclarationInstantiator(parse.ASTVisitor):
    def __init__(self, rt):
        self.env = rt.context.lex_env.record

    def visit_variable_decl(self, decl):
        name = decl.name
        if not self.env.has_binding(name):
            self.env.create_mutable_binding(name, False)
            self.env.set_mutable_binding(name, Undefined, False)


def instantiate_declaration_binding(rt, code, fn_info=None):
    code_is_function = code.__class__ is parse.FunctionDecl
    if code_is_function:
        assert fn_info is not None

    env = rt.context.lex_env.record
    # Step 4.
    if code_is_function:
        f_obj, f_args = fn_info
        for arg_name, arg_value in zip(f_obj.formal_parameters, f_args):
            if not env.has_binding(arg_name):
                env.create_mutable_binding(arg_name, False)
            env.set_mutable_binding(arg_name, arg_value, False)

    # Step 5.
    v = FunctionDeclarationInstantiator(rt)
    if not code_is_function:
        code.accept(v)
    else:
        for element in code.body:
            element.accept(v)

    # Step 6. 7.
    # Step 8.
    v = VariableDeclarationInstantiator(rt)
    code.accept(v)

class Evaluator(parse.ASTVisitor):
    EMPTY, BREAK, CONTINUE = range(3)

    def __init__(self, rt, root):
        self.rt = rt
        self.root = root
        self.rval = Undefined
        self.type = Evaluator.EMPTY

    def visit_program(self, program):
        assert program == self.root

        if not program.elements:
            return

        global_env = new_object_environment(self.rt.global_object, None)
        context = ExecutionContext(global_env, global_env,
                                   self.rt.global_object)
        self.rt.push_context(context)
        self.rt.global_env = global_env 

        instantiate_declaration_binding(self.rt, program)

        for elem in program.elements:
            if elem.__class__ is parse.FunctionDecl:
                continue
            elem.accept(self)

        self.rt.pop_context()

    def visit_function_decl(self, decl):
        assert decl == self.root
        for elem in decl.body:
            if elem.__class__ is parse.FunctionDecl:
                continue
            elem.accept(self)

    # 12.2 Variable Statement
    def visit_variable_decl(self, decl):
        if not decl.initialiser:
            return
        lhs = get_identifier_reference(self.rt.context.lex_env,
                                       decl.name, False)
        decl.initialiser.accept(self)
        rhs = self.rval
        put_value(self.rt, lhs, get_value(rhs))

    def visit_for_stmt_with_var_decl(self, stmt):
        for decl in stmt.decls:
            decl.accept(self)
        while True:
            if stmt.test_expr:
                stmt.test_expr.accept(self)
                if not to_boolean(get_value(self.rval)):
                    return
            stmt.stmt.accept(self)
            if stmt.inc_expr:
                stmt.inc_expr.accept(self)
                get_value(self.rval)

    def visit_return_stmt(self, stmt):
        stmt.expr.accept(self)
        raise Return(self.rval)

    def visit_expr_statement(self, statement):
        statement.expr.accept(self)

    def visit_expr(self, expr):
        exprs = expr.exprs
        for i in range(len(exprs) - 1):
            exprs[i].accept(self)
            get_value(self.rval)
        exprs[-1].accept(self)
        self.rval = get_value(self.rval)

    def visit_assignment_expr(self, expr):
        expr.lhs.accept(self)
        lref = self.rval
        expr.rhs.accept(self)
        rval = get_value(self.rval)
        if expr.op == parse.AssignmentExpr.ASSIGN:
            pass
        elif expr.op == parse.AssignmentExpr.ADDASSIGN:
            rval = get_value(lref) + rval
        elif expr.op == parse.AssignmentExpr.MULASSIGN:
            rval = get_value(lref) * rval
        put_value(self.rt, lref, rval)
        self.rval = rval

    def visit_binary_op_expr(self, expr):
        expr.lhs.accept(self)
        lval = get_value(self.rval)
        expr.rhs.accept(self)
        rval = get_value(self.rval)

        if expr.op == parse.BinaryOpExpr.ADD:
            self.rval = (to_number(to_primitive(lval)) +
                         to_number(to_primitive(rval)))
        elif expr.op == parse.BinaryOpExpr.SUB:
            self.rval = to_int32(lval) - to_int32(rval)
        elif expr.op == parse.BinaryOpExpr.MUL:
            self.rval = lval * rval
        elif expr.op == parse.BinaryOpExpr.LT:
            self.rval = lval < rval
        elif expr.op == parse.BinaryOpExpr.GT:
            self.rval = lval > rval
        elif expr.op == parse.BinaryOpExpr.BITOR:
            self.rval = to_int32(lval) | to_int32(rval)
        elif expr.op == parse.BinaryOpExpr.BITAND:
            self.rval = to_int32(lval) & to_int32(rval)
        elif expr.op == parse.BinaryOpExpr.BITXOR:
            self.rval = to_int32(lval) ^ to_int32(rval)
        elif expr.op == parse.BinaryOpExpr.LSH:
            self.rval = to_int32(lval) << to_uint32(rval) & 0x1f
        elif expr.op == parse.BinaryOpExpr.RSH:
            self.rval = to_int32(lval) >> to_uint32(rval) & 0x1f
        else:
            raise NotImplementedError("Unknown operator!")

    def visit_unary_expr(self, expr):
        if expr.op == parse.UnaryExpr.INC:
            expr.expr.accept(self)
            sub_expr = self.rval
            old_value = to_number(get_value(sub_expr))
            self.rval = old_value + 1
            put_value(self.rt, sub_expr, self.rval)
        elif expr.op == parse.UnaryExpr.MINUS:
            expr.expr.accept(self)
            old_value = to_number(get_value(self.rval))
            if old_value == NaN:
                return NaN
            self.rval = -old_value
        elif expr.op == parse.UnaryExpr.BITNOT:
            expr.expr.accept(self)
            self.rval = ~to_int32(get_value(self.rval))

    def visit_postfix_expr(self, expr):
        expr.expr.accept(self)
        # FIXME Step 2.
        old_value = to_number(get_value(self.rval))
        put_value(self.rt, self.rval, old_value + 1.0)
        self.rval = old_value

    # 11.2.3 Function Calls
    def visit_call_expr(self, expr):
        expr.callee.accept(self)
        func = get_value(self.rval)
        arg_list = []
        for arg_node in expr.args:
            arg_node.accept(self)
            arg_list.append(get_value(self.rval))
        if not isinstance(func, Object):
            raise ESTypeError
        if not is_callable(func):
            raise ESTypeError

        if isinstance(self.rval, Reference):
            if self.rval.is_property_reference():
                this_value = self.rval.base
            else:
                this_value = self.rval.base.implicit_this_value()
        else:
            this_value = Undefined
        self.rval = func.call(self.rt, this_value, arg_list)

    def visit_dot_property_access(self, access):
        access.expr.accept(self)
        base_value = get_value(self.rval)
        check_object_coercible(base_value)
        self.rval = Reference(base_value, access.name, False)

    # 10.3.1 Identifier Resolution
    def visit_identifier(self, ident):
        self.rval = get_identifier_reference(self.rt.context.lex_env,
                                             ident.name, False)

    def visit_numeric_literal(self, literal):
        self.rval = literal.value

    def visit_this(self, this):
        self.rval = self.rt.context.this


class Runtime:
    def __init__(self):
        self.context_stack = []
        self.global_object = None

    def push_context(self, context):
        self.context_stack.append(context)

    def pop_context(self):
        self.context_stack.pop()

    @property
    def context(self):
        return self.context_stack[-1]

    def set_global_object(self, glob):
        self.global_object = glob


def execute(rt, src):
    ast = parse.parse(src)
    v = Evaluator(rt, ast)
    ast.accept(v)
