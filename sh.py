# -*- coding: utf-8 -*-

"The ECMAScript Shell"

import argparse
import os.path
import sys
import time

import es

# --- Shell -----------------------------------------------------------------

def fn_print(this, *args):
    print(*args)

def fn_assertEq(this, actual, expected):
    if actual == expected:
        print("  assertEq: OK!")
    else:
        msg = "  assertEq: FAIL! --- expected {} but found {}!"
        print(msg.format(expected, actual))

def new_global_object(rt):
    global_object = es.GlobalObject()

    obj = es.NativeFunction(None, fn_print, False)
    desc = es.PropertyDescriptor(value=obj)
    assert global_object.define_own_property("print", desc, False)
    assert global_object.has_property("print")

    obj = es.NativeFunction(None, fn_assertEq, False)
    desc = es.PropertyDescriptor(value=obj)
    global_object.define_own_property("assertEq", desc, False)

    return global_object

def cmd_scan(args):
    scan = parse.Scanner(args.source)
    while True:
        token = scan.next_token()
        print(token)
        if token.kind == parse.Token.EOF:
            break

def get_source(args):
    path = args.source
    if not os.path.exists(path):
        alter_path = "jit-test\\tests\\basic\\" + path
        if not os.path.exists(alter_path):
            exit("can't find file '{}'".format(path))
        path = alter_path
    with open(path) as fp:
        return fp.read()

def cmd_parse(args):
    import parse

    source = get_source(args)
    ast = parse.parse(source)
    visitor = parse.XMLDumpVisitor()
    print("<?xml version=\"1.0\" encoding=\"utf-8\"?>")
    ast.accept(visitor)

def cmd_execute(args):
    src = get_source(args)

    rt = es.Runtime()
    glob = new_global_object(rt)
    rt.set_global_object(glob)

    if not args.profile:
        es.execute(rt, src)
    else:
        import cProfile as profile
        profile.runctx("es.execute(rt, src)", globals(), locals())

tests = r"""
basic\testBitwise.js
basic\arityMismatchExtraArg.js
basic\arityMismatchMissingArg.js
basic\bug566637.js
basic\globalGet.js
basic\globalSet.js
basic\setCall.js
basic\name.js
sunspider\check-bitops-bitwise-and.js
sunspider\check-bitops-3bit-bits-in-byte.js
basic\bitwiseAnd.js
"""

def cmd_test(args):
    for test in tests.strip().split("\n"):
        if test[0] == "#":
            continue

        print(test)
        with open("jit-test\\tests\\" + test) as f:
            src = f.read()
        rt = es.Runtime()
        rt.set_global_object(new_global_object(rt))

        t = time.time()
        es.execute(rt, src)
        print("  (Elapsed time: {:5.3f} sec)".format(time.time() - t))

def main():
    arg_parser = argparse.ArgumentParser()
    subs = arg_parser.add_subparsers()

    sub = subs.add_parser("test", help="run tests")
    sub.set_defaults(function=cmd_test)

    sub = subs.add_parser("execute", help="execute a script")
    sub.add_argument("source")
    sub.add_argument("-p", "--profile", action="store_true", default=False,
                     help="activate the python profiler (cProfile)")
    sub.set_defaults(function=cmd_execute)

    sub = subs.add_parser("parse",
                          help="parse a script and show the syntax tree as xml")
    sub.add_argument("source")
    sub.set_defaults(function=cmd_parse)

    sub = subs.add_parser("scan", help="scan a script and list tokens")
    sub.add_argument("source")
    sub.set_defaults(function=cmd_scan)

    args = arg_parser.parse_args(sys.argv[1:])
    args.function(args)

    exit()

if __name__ == "__main__":
    main()
