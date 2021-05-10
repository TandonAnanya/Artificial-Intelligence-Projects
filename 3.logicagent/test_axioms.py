import sys
import time
import traceback

import logic
from axiom_testcases import axiom_tests

def error_method_not_implemented(name):
    print("*** Method not implemented: %s".format(name))

def print_test(num, test, name, axiom, inferTrue, inferFalse, succeed):
    if succeed: print("*** PASS: test_%d" % (num + 1))
    else: print("*** FAIL: test_%d" % (num + 1))
    if not axiom: return
    print("***      Input:  {0}({1})".format(
        name,
        ', '.join([str(x) for x in test['input']])
    ))
    print("***      Your axiom return: {0}".format(axiom))
    print("***      Test propositions: ({0})({1})".format(
        [x for x in test.get('inferTrue', [])],
        [x for x in test.get('inferFalse', [])]
    ))
    print("***      Expected truth value: ({0})({1})".format(
        [True for x in inferTrue],
        [False for x in inferFalse]
    ))
    print("***      Your output: ({0})({1})".format(
        [x for x in inferTrue],
        [x for x in inferFalse]
    ))

def test_axiom_fn(name, callable, tests, score):
    test_score = score
    print(name)
    print("=================================================\n")

    for i, test in enumerate(tests):
        try:
            axiom = callable(*test['input'])
            if not axiom:
                error_method_not_implemented(name)
                test_score = 0
                print_test(i, test, name, None, [], [], False)
            else:
                inferTrue = [
                    logic.tt_true(logic.expr("({0}) >> ({1})".format(
                        axiom, x
                    )))
                    for x in test.get('inferTrue', [])
                ]
                inferFalse = [
                    logic.tt_true(logic.expr("({0}) >> ({1})".format(
                        axiom, x
                    )))
                    for x in test.get('inferFalse', [])
                ]
                if all(inferTrue) and not any(inferFalse):
                    print_test(i, test, name, axiom, inferTrue,inferFalse, True)
                else:
                    test_score = 0
                    print_test(i, test, name, axiom, inferTrue, inferFalse, False)

        except Exception as ex:
            print('Exception raised: %s' % ex)
            for line in traceback.format_exc().split('\n'):
                print(line)
            test_score = 0
            print_test(i, test, name, None, [], [], False)

        print('')

    print("\n### {0}: {1}/{2}\n".format(name, test_score, score))
    return test_score

def test_all_axioms():
    print('Starting on %d-%d at %d:%02d:%02d\n' % time.localtime()[1:6])
    print("Testing axiom functions")
    print("=========================\n")

    for test in axiom_tests:
        score = test_axiom_fn(
            test['name'], 
            test['callable'], 
            test.get('tests', []), 
            test.get('score', 0)
        )
        test['received'] = score
    
    print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
    print("\nProvisional grades\n==================")

    for test in axiom_tests:
        print("{0: <55}: {1}/{2}".format(
            test['name'], test['received'], test['score']
        ))

    total = sum([test['received'] for test in axiom_tests])

    print('------------------')
    print('Total: {0}/24.0'.format(total))
    print('------------------\n')

def test_single_axiom_fn(fn):
    for test in axiom_tests:
        if test['name'] == fn:
            score = test_axiom_fn(
                test['name'], 
                test['callable'], 
                test.get('tests', []), 
                test.get('score', 0)
            )
            return

    print("The specified axiom function '{0}' is invalid".format(fn))

#-------------------------------------------------------------------------------
# Command-line interface
#-------------------------------------------------------------------------------

def default(str):
    return str + ' [Default: %default]'

def readCommand(argv):
    """
    Processes the command used to run test_axioms.py from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:     python test_axioms.py <options>
    EXAMPLES:  (1) python test_axioms.py
                   - tests all the axiom functions
               (2) python test_axioms.py -f OR python test_axioms.py --fn
                   - tests a particular axiom function
    """
    parser = OptionParser(usageStr)
    parser.add_option('-f', '--fn', dest='test_fn', default=False,
                      help=default("Test a particular axiom function"))

    options, otherjunk = parser.parse_args(argv)
    
    if len(otherjunk) != 0:
        raise Exception("Command line input not understood: " + str(otherjunk))

    return options

def run_command(options):
    if options.test_fn:
        test_single_axiom_fn(options.test_fn)
    else:
        test_all_axioms()

if __name__ == "__main__":
    options = readCommand(sys.argv[1:])
    run_command(options)
