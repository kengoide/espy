# -*- coding: utf-8 -*-

from copy import copy
import string
import sys

class ParseError(Exception):
    def __init__(self, line, column, msg):
        self.line = line
        self.column = column
        self.msg = msg

    def __str__(self):
        return "({}, {}): {}".format(self.line, self.column, self.msg)

# --- Scanner ----------------------------------------------------------------

_token_map = [
    ("EOF", "[EOF]"),
    ("LPAREN", "("),
    ("RPAREN", ")"),
    ("LBRACE", "{"),
    ("RBRACE", "}"),
    ("DOT", "."),
    ("COMMA", ","),
    ("SEMICOLON", ";"),
    ("IDENT", "<identifier>"),
    ("NUMBER", "<number>"),
    ("LSH", "<<"),
    ("RSH", ">>"),
    ("URSH", ">>>"),
    ("PLUS", "+"),
    ("MINUS", "-"),
    ("MUL", "*"),
    ("BITOR", "|"),
    ("BITAND", "&"),
    ("BITXOR", "^"),
    ("LT", "<"),
    ("GT", ">"),
    ("ASSIGN", "="),
    ("ADDASSIGN", "+="),
    ("MULASSIGN", "*="),
    ("INC", "++"),
    ("DEC", "--"),
    ("BITNOT", "~"),
    ("FOR", "for"),
    ("FUNCTION", "function"),
    ("RETURN", "return"),
    ("THIS", "this"),
    ("VAR", "var"),
]

def _make_token_class():
    class _Token:
        _reprs = []

        def __init__(self, kind, data=None, source=None):
            self.kind = kind
            self.data = data
            self.source = source

        def __eq__(self, other):
            return self.kind == other.kind and self.data == other.data

        def __str__(self):
            if self.source:
                return self.source
            return _Token._reprs[self.kind]

    for i, (token_name, token_repr) in enumerate(_token_map):
        setattr(_Token, token_name, i)
        _Token._reprs.append(token_repr)
    return _Token

Token = _make_token_class()
del _make_token_class

_keywords = [Token.FUNCTION, Token.FOR, Token.RETURN, Token.THIS, Token.VAR]
_keywords = {_token_map[n][1]: n for n in _keywords}

del _token_map

class Scanner:
    def __init__(self, src):
        self.src = src
        self.curr = None
        self.pos = 0
        self.line = 1
        self.column = 1
        self.peek_tok = Token(Token.EOF)
        if src:
            self.curr = self.src[self.pos]
            self.pos += 1

        self._look_ahead()

    def next_token(self):
        ret = self.peek_tok
        self._look_ahead()
        return ret

    def is_eof(self):
        return not self.curr

    def _look_ahead(self):
        if self.is_eof():
            self.peek_tok = Token(Token.EOF)
        else:
            self.peek_tok = self._analyze()

    def _shift(self):
        if self.pos < len(self.src):
            if self.curr == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.curr = self.src[self.pos]
            self.pos += 1
        else:
            if self.curr:
                self.curr = "\0"

    _line_terminators = "\u000a\u000d\u2028\u2029"

    def _analyze(self):
        accum = ""
        c = self.curr

        while True:
            while c.isspace():
                self._shift()
                c = self.curr

            if c == "/":
                self._shift()
                if self.curr == "/":
                    self._shift()
                    while not self.curr in Scanner._line_terminators:
                        self._shift()
                elif self.curr == "*":
                    self._shift()
                    while True:
                        while self.curr != "*":
                            self._shift()
                        self._shift()
                        if self.curr == "/":
                            break
                        self._shift()
                    self._shift()
            else:
                break
            c = self.curr

        if c.isalpha() or c == "_":
            while c.isalnum() or c == "_":
                accum += c
                self._shift()
                c = self.curr

            maybe_keyword = _keywords.get(accum)
            if maybe_keyword:
                return Token(maybe_keyword)
            return Token(Token.IDENT, accum, accum)

        # 7.8.3 NumericLiteral
        if c == "0":
            self._shift()
            if self.curr == "x" or self.curr == "X":
                accum += self.curr
                self._shift()
                while self.curr in "0123456789ABCDEFabcdef":
                    accum += self.curr
                    self._shift()
                if len(accum) < 2:
                    self._raise("Invalid syntax!")
                return Token(Token.NUMBER, float(int(accum[1:], 16)),
                             "0" + accum)
            if self.curr in "0123456789":
                self._raise("Decimal literals must begin with non-zero!")
            return Token(Token.NUMBER, +0.0, "0")
        if c in "123456789":
            while c in "0123456789":
                accum += c
                self._shift()
                c = self.curr
            return Token(Token.NUMBER, float(accum), accum)

        if c == "(":
            self._shift()
            return Token(Token.LPAREN)
        elif c == ")":
            self._shift()
            return Token(Token.RPAREN)
        elif c == "{":
            self._shift()
            return Token(Token.LBRACE)
        elif c == "}":
            self._shift()
            return Token(Token.RBRACE)
        elif c == ".":
            self._shift()
            return Token(Token.DOT)
        elif c == ",":
            self._shift()
            return Token(Token.COMMA)
        elif c == ";":
            self._shift()
            return Token(Token.SEMICOLON)
        elif c == "+":
            self._shift()
            if self.curr == "+":
                self._shift()
                return Token(Token.INC)
            if self.curr == "=":
                self._shift()
                return Token(Token.ADDASSIGN)
            return Token(Token.PLUS)
        elif c == "-":
            self._shift()
            if self.curr == "-":
                self._shift()
                return Token(Token.DEC)
            return Token(Token.MINUS)
        elif c == "*":
            self._shift()
            if self.curr == "=":
                self._shift()
                return Token(Token.MULASSIGN)
            return Token(Token.MUL)
        elif c == "=":
            self._shift()
            return Token(Token.ASSIGN)
        elif c == "<":
            self._shift()
            if self.curr == "<":
                self._shift()
                return Token(Token.LSH)
            return Token(Token.LT)
        elif c == ">":
            self._shift()
            if self.curr == ">":
                self._shift()
                if self.curr == ">":
                    self.shift()
                    return Token(Token.URSH)
                return Token(Token.RSH)
            return Token(Token.GT)
        elif c == "|":
            self._shift()
            return Token(Token.BITOR)
        elif c == "&":
            self._shift()
            return Token(Token.BITAND)
        elif c == "^":
            self._shift()
            return Token(Token.BITXOR)
        elif c == "~":
            self._shift()
            return Token(Token.BITNOT)
        elif c == "\0":
            return Token(Token.EOF)
        else:
            self._raise("not implemented yet!")

    def _raise(self, message):
        raise ParseError(self.line, self.column, message)


# --- Parser -----------------------------------------------------------------

class _NodeMeta(type):
    _decamel_table = {ord(c): "_" + c.lower() for c in string.ascii_uppercase}

    def __new__(mcs, name, bases, dic):
        visit_method_name = "visit" + name.translate(_NodeMeta._decamel_table)
        def accept(self, v):
            getattr(v, visit_method_name)(self)
        dic["accept"] = accept
        return super().__new__(mcs, name, bases, dic)

class _Node(metaclass=_NodeMeta):
    pass

class NumericLiteral(_Node):
    def __init__(self, value):
        self.value = value

class Identifier(_Node):
    def __init__(self, name):
        self.name = name

class This(_Node):
    pass

class DotPropertyAccess(_Node):
    def __init__(self, expr, name):
        self.expr = expr
        self.name = name

class CallExpr(_Node):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

class PostfixExpr(_Node):
    INC = 0
    DEC = 1

    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class UnaryExpr(_Node):
    INC, MINUS, BITNOT = range(3)

    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class BinaryOpExpr(_Node):
    ADD, SUB, MUL, LT, GT, BITOR, BITAND, BITXOR, LSH, RSH, URSH = range(11)

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class AssignmentExpr(_Node):
    ASSIGN, ADDASSIGN, MULASSIGN = range(3)

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

class Expr(_Node):
    def __init__(self, exprs):
        self.exprs = exprs

class ExprStmt(_Node):
    def __init__(self, expr):
        self.expr = expr

class ReturnStmt(_Node):
    def __init__(self, expr):
        self.expr = expr

class ForStmtWithVarDecl(_Node):
    def __init__(self, decls, test_expr, inc_expr, stmt):
        self.decls = decls
        self.test_expr = test_expr
        self.inc_expr = inc_expr
        self.stmt = stmt

class VariableStmt(_Node):
    def __init__(self, decls):
        self.decls = decls

class Block(_Node):
    def __init__(self, stmts):
        self.stmts = stmts

class VariableDecl(_Node):
    def __init__(self, name, initialiser):
        self.name = name
        self.initialiser = initialiser

class FunctionDecl(_Node):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class Program(_Node):
    def __init__(self, elements):
        self.elements = elements

class _Parser:
    def __init__(self, scanner):
        self.scan = scanner
        self.token = scanner.next_token()

    def consume(self, allow_lineterm=True):
        self.token = self.scan.next_token()

    def snapshot(self):
        return (copy(self.scan), copy(self.token))

    def restore(self, snapshot):
        self.scan, self.token = snapshot

    def expect(self, token):
        if token == self.token.kind:
            self.consume()
            return
        elif token == Token.SEMICOLON and self.token.kind == Token.EOF:
            # XXX: ad-hoc semicolon insertion!
            return
        expected = Token(token)
        msg = "expected `{0}` but found `{1}`".format(expected, self.token)
        self.raise_error(msg)

    def raise_error(self, message):
        raise ParseError(self.scan.line, self.scan.column, message)
    
    def parse_program(self):
        return Program(self.parse_source_elements(Token.EOF))

    def parse_source_elements(self, term):
        elems = []
        while self.token.kind != term:
            if self.token.kind == Token.FUNCTION:
                self.consume()
                elems.append(self.parse_function_decl())
            else:
                elems.append(self.parse_stmt())
        return elems

    def parse_function_decl(self):
        name = self.token.data
        self.expect(Token.IDENT)
        self.expect(Token.LPAREN)

        params = []
        first = True
        while self.token.kind != Token.RPAREN:
            if not first:
                self.expect(Token.COMMA)
            param_name = self.token.data
            self.expect(Token.IDENT)
            params.append(param_name)
        self.consume()
        self.expect(Token.LBRACE)
        
        if self.token.kind == Token.RBRACE:
            body = []
        else:
            body = self.parse_source_elements(Token.RBRACE)
        self.consume()

        return FunctionDecl(name, params, body)

    def parse_stmt(self):
        if self.token.kind == Token.LBRACE:
            self.consume()
            stmts = []
            while self.token.kind != Token.RBRACE:
                stmts.append(self.parse_stmt())
            self.expect(Token.RBRACE)
            return Block(stmts)
        if self.token.kind == Token.VAR:
            self.consume()
            return self.parse_variable_stmt()
        if self.token.kind == Token.FOR:
            self.consume()
            return self.parse_for_stmt()
        if self.token.kind == Token.RETURN:
            self.consume()
            if self.token.kind == Token.SEMICOLON:
                self.consume()
                expr = None
            else:
                expr = self.parse_expr()
                self.expect(Token.SEMICOLON)
            return ReturnStmt(expr)
        retval = self.parse_expr_stmt()
        return retval

    def parse_variable_stmt(self):
        decls = self.parse_variable_decl_list()
        self.expect(Token.SEMICOLON)
        return VariableStmt(decls)

    def parse_variable_decl_list(self):
        decls = [self.parse_variable_decl()]
        while self.token.kind == Token.COMMA:
            self.consume()
            decls.append(self.parse_variable_decl())
        return decls

    def parse_variable_decl(self):
        if self.token.kind != Token.IDENT:
            msg = "expected identifier but found `" + str(self.token) + "`"
            self.raise_error(msg)
        name = self.token.data
        self.consume()
        if self.token.kind == Token.ASSIGN:
            self.consume()
            initialiser = self.parse_assignment_expr()
        else:
            initialiser = None
        return VariableDecl(name, initialiser)

    def parse_for_stmt(self):
        self.expect(Token.LPAREN)
        self.expect(Token.VAR)
        var_decls = self.parse_variable_decl_list()
        self.expect(Token.SEMICOLON)

        if self.token.kind == Token.SEMICOLON:
            self.consume()
            test_expr = None
        else:
            test_expr = self.parse_expr()
            self.expect(Token.SEMICOLON)

        if self.token.kind == Token.RPAREN:
            self.consume()
            inc_expr = None
        else:
            inc_expr = self.parse_expr()
            self.expect(Token.RPAREN)

        stmt = self.parse_stmt()
        return ForStmtWithVarDecl(var_decls, test_expr, inc_expr, stmt)

    def parse_expr_stmt(self):
        expr = self.parse_expr()
        self.expect(Token.SEMICOLON)
        return ExprStmt(expr)

    def parse_expr(self):
        exprs = [self.parse_assignment_expr()]
        while self.token.kind == Token.COMMA:
            self.consume()
            exprs.append(self.parse_assignment_expr())
        return Expr(exprs)

    def parse_assignment_expr(self):
        snapshot = self.snapshot()
        lhs = self.parse_lhs_expr()
        if lhs:
            if self.token.kind == Token.ASSIGN:
                self.consume()
                return AssignmentExpr(AssignmentExpr.ASSIGN, lhs,
                                      self.parse_assignment_expr())
            elif self.token.kind == Token.ADDASSIGN:
                self.consume()
                return AssignmentExpr(AssignmentExpr.ADDASSIGN, lhs,
                                      self.parse_assignment_expr())
            elif self.token.kind == Token.MULASSIGN:
                self.consume()
                return AssignmentExpr(AssignmentExpr.MULASSIGN, lhs,
                                      self.parse_assignment_expr())

        self.restore(snapshot)
        return self.parse_bitwise_or_expr()

    def parse_bitwise_or_expr(self):
        expr = self.parse_bitwise_xor_expr()
        while self.token.kind == Token.BITOR:
            self.consume()
            expr = BinaryOpExpr(BinaryOpExpr.BITOR, expr,
                                self.parse_bitwise_xor_expr())
        return expr

    def parse_bitwise_xor_expr(self):
        expr = self.parse_bitwise_and_expr()
        while self.token.kind == Token.BITXOR:
            self.consume()
            expr = BinaryOpExpr(BinaryOpExpr.BITXOR, expr,
                                self.parse_bitwise_and_expr())
        return expr

    def parse_bitwise_and_expr(self):
        expr = self.parse_relational_expr()
        while self.token.kind == Token.BITAND:
            self.consume()
            expr = BinaryOpExpr(BinaryOpExpr.BITAND, expr,
                                self.parse_relational_expr())
        return expr

    relational_op_tokens = [Token.LT, Token.GT]

    def parse_relational_expr(self):
        expr = self.parse_shift_expr()
        while self.token.kind in _Parser.relational_op_tokens:
            if self.token.kind == Token.LT:
                op = BinaryOpExpr.LT
            else:
                op = BinaryOpExpr.GT
            self.consume()
            expr = BinaryOpExpr(op, expr, self.parse_shift_expr())
        return expr

    shift_op_tokens = [Token.LSH, Token.RSH, Token.URSH]

    def parse_shift_expr(self):
        expr = self.parse_additive_expr()
        while self.token.kind in _Parser.shift_op_tokens:
            if self.token.kind == Token.LSH:
                op = BinaryOpExpr.LSH
            elif self.token.kind == Token.RSH:
                op = BinaryOpExpr.RSH
            elif self.token.kind == Token.URSH:
                op = BinaryOpExpr.URSH
            else:
                raise Exception("wtf!?")
            self.consume()
            expr = BinaryOpExpr(op, expr, self.parse_additive_expr())
        return expr

    def parse_additive_expr(self):
        expr = self.parse_multiplicative_expr()
        while self.token.kind == Token.PLUS or self.token.kind == Token.MINUS:
            if self.token.kind == Token.PLUS:
                op = BinaryOpExpr.ADD
            else:
                op = BinaryOpExpr.SUB
            self.consume()
            expr = BinaryOpExpr(op, expr, self.parse_multiplicative_expr())
        return expr

    _unary_ops = {Token.INC: UnaryExpr.INC, Token.MINUS: UnaryExpr.MINUS,
                  Token.BITNOT: UnaryExpr.BITNOT}

    def parse_multiplicative_expr(self):
        expr = self.parse_unary_expr()
        while self.token.kind == Token.MUL:
            self.consume()
            expr = BinaryOpExpr(BinaryOpExpr.MUL, expr,
                                self.parse_unary_expr())
        return expr

    def parse_unary_expr(self):
        op = self.__class__._unary_ops.get(self.token.kind)
        if op is not None:
            self.consume()
            return UnaryExpr(op, self.parse_unary_expr())
        return self.parse_postfix_expr()

    def parse_postfix_expr(self):
        expr = self.parse_lhs_expr()
        if self.token.kind == Token.INC:
            self.consume()
            return PostfixExpr(PostfixExpr.INC, expr)
        elif self.token.kind == Token.DEC:
            self.consume()
            return PostfixExpr(PostfixExpr.DEC, expr)
        return expr

    def parse_lhs_expr(self):
        expr = self.parse_member_expr()
        if self.token.kind == Token.LPAREN:
            return self.parse_call_expr(expr)
        return expr

    def parse_call_expr(self, callee):
        args = self.parse_arguments()
        return CallExpr(callee, args)

    def parse_arguments(self):
        self.expect(Token.LPAREN)

        if self.token.kind == Token.RPAREN:
            self.consume()
            return []

        args = self.parse_argument_list()
        self.expect(Token.RPAREN)
        return args

    def parse_argument_list(self):
        exprs = [self.parse_assignment_expr()]
        while self.token.kind == Token.COMMA:
            self.consume()
            exprs.append(self.parse_assignment_expr())
        return exprs

    def parse_member_expr(self):
        expr = self.parse_primary_expr()
        while self.token.kind == Token.DOT:
            self.consume()
            name = self.token.data
            self.expect(Token.IDENT)
            expr = DotPropertyAccess(expr, name)
        return expr

    def parse_primary_expr(self):
        if self.token.kind == Token.THIS:
            self.consume()
            return This()
        if self.token.kind == Token.IDENT:
            ident = self.token.data
            self.consume()
            return Identifier(ident)
        if self.token.kind == Token.NUMBER:
            value = self.token.data
            self.consume()
            return NumericLiteral(value)
        if self.token.kind == Token.LPAREN:
            self.consume()
            expr = self.parse_expr()
            self.expect(Token.RPAREN)
            return expr
        return None


def parse(source):
    scanner = Scanner(source)
    parser = _Parser(scanner)
    return parser.parse_program()

class ASTVisitor:
    def visit_program(self, program):
        for elem in program.elements:
            elem.accept(self)

    def visit_function_decl(self, decl):
        for stmt in decl.body:
            stmt.accept(self)

    def visit_block(self, block):
        for stmt in block.stmts:
            stmt.accept(self)

    def visit_return_stmt(self, stmt):
        if stmt.expr:
            stmt.expr.accept(self)

    def visit_variable_stmt(self, stmt):
        for decl in stmt.decls:
            decl.accept(self)

    def visit_variable_decl(self, decl):
        if decl.initialiser:
            decl.initialiser.accept(self)

    def visit_for_stmt_with_var_decl(self, stmt):
        for decl in stmt.decls:
            decl.accept(self)
        stmt.test_expr.accept(self)
        stmt.inc_expr.accept(self)
        stmt.stmt.accept(self)

    def visit_expr_stmt(self, expr_stmt):
        expr_stmt.expr.accept(self)

    def visit_expr(self, expr):
        for sub_expr in expr.exprs:
            sub_expr.accept(self)

    def visit_assignment_expr(self, expr):
        expr.lhs.accept(self)
        expr.rhs.accept(self)

    def visit_binary_op_expr(self, expr):
        expr.lhs.accept(self)
        expr.rhs.accept(self)

    def visit_unary_expr(self, expr):
        expr.expr.accept(self)

    def visit_postfix_expr(self, expr):
        expr.expr.accept(self)

    def visit_call_expr(self, call_expr):
        call_expr.callee.accept(self)
        for arg in call_expr.args:
            arg.accept(self)

    def visit_dot_property_access(self, access):
        access.expr.accept(self)

    def visit_identifier(self, ident):
        pass

    def visit_numeric_literal(self, num_literal):
        pass

    def visit_this(self, this):
        pass


from contextlib import contextmanager

@contextmanager
def tag(arg, **attrs):
    if isinstance(arg, str):
        tagname = arg
    else:
        tagname = arg.__class__.__name__
    sys.stdout.write("<" + tagname)
    for (attrname, value) in attrs.items():
        sys.stdout.write(" " + attrname + "=\"" + str(value) + "\"")
    sys.stdout.write(">")
    yield
    sys.stdout.write("</" + tagname + ">")

class XMLDumpVisitor(ASTVisitor):
    def visit_program(self, program):
        with tag(program):
            for elem in program.elements:
                elem.accept(self)

    def visit_function_decl(self, decl):
        with tag(decl, Name=decl.name, FormalParameters=",".join(decl.params)):
            for stmt in decl.body:
                stmt.accept(self)

    def visit_block(self, stmt):
        with tag(stmt):
            for s in stmt.stmts:
                s.accept(self)

    def visit_variable_stmt(self, stmt):
        with tag(stmt):
            for decl in stmt.decls:
                decl.accept(self)

    def visit_variable_decl(self, decl):
        with tag(decl, Name=decl.name):
            if decl.initialiser:
                decl.initialiser.accept(self)

    def visit_for_stmt_with_var_decl(self, stmt):
        with tag(stmt):
            with tag("VariableDeclarationList"):
                for decl in stmt.decls:
                    decl.accept(self)
            stmt.test_expr.accept(self)
            stmt.inc_expr.accept(self)
            stmt.stmt.accept(self)

    def visit_return_stmt(self, stmt):
        with tag(stmt):
            if stmt.expr:
                stmt.expr.accept(self)

    def visit_expr_stmt(self, expr_stmt):
        with tag(expr_stmt):
            expr_stmt.expr.accept(self)

    def visit_expr(self, expr):
        with tag(expr):
            for sub_expr in expr.exprs:
                sub_expr.accept(self)

    _assignment_expr_repl = {AssignmentExpr.ASSIGN: "="}

    def visit_assignment_expr(self, expr):
        op_s = self.__class__._assignment_expr_repl[expr.op]
        with tag(expr, Op=op_s):
            expr.lhs.accept(self)
            expr.rhs.accept(self)

    _bin_op_repl = ["+", "-", "&lt;", "&gt;", "|", "&amp;", "^"]

    def visit_binary_op_expr(self, expr):
        op_repr = self.__class__._bin_op_repl[expr.op]
        with tag(expr, Op=op_repr):
            expr.lhs.accept(self)
            expr.rhs.accept(self)

    _unary_repl = ["-", "~"]

    def visit_unary_expr(self, expr):
        op_repr = self.__class__._unary_repl[expr.op]
        with tag(expr, Op=op_repr):
            expr.expr.accept(self)

    _postfix_repl = {PostfixExpr.INC: "++", PostfixExpr.DEC: "--"}

    def visit_postfix_expr(self, expr):
        op_repr = self.__class__._postfix_repl[expr.op]
        with tag(expr, Op=op_repr):
            expr.expr.accept(self)

    def visit_call_expr(self, call_expr):
        with tag(call_expr):
            call_expr.callee.accept(self)
            for arg in call_expr.args:
                arg.accept(self)

    def visit_dot_property_access(self, expr):
        with tag(expr, Name=expr.name):
            expr.expr.accept(self)

    def visit_identifier(self, ident):
        with tag(ident, Name=ident.name):
            pass

    def visit_numeric_literal(self, num_literal):
        with tag(num_literal, Value=num_literal.value):
            pass

    def visit_this(self, this):
        with tag(this):
            pass
