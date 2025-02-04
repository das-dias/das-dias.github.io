This is a collection of interesting tools, sandboxes and projects I end up doing when I want to learn new things about programming (and more specifically scientific-programming), IC Design how EDA tools for IC design work and can be innovated.

Hope you find something that you would like to try for yourself!  

# Programming in C-lang with Jupyter Notebooks

## Introduction: Hello World!

Start by writting your C-lang file:


```python

%%file hello_world.c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  printf("Hello World from C in Jupyter!");
  exit(0);
}
```

    Overwriting hello_world.c


Now, compile your C code.


```bash
%%bash
gcc --version
```

    Apple clang version 15.0.0 (clang-1500.3.9.4)
    Target: arm64-apple-darwin23.6.0
    Thread model: posix
    InstalledDir: /Library/Developer/CommandLineTools/usr/bin


Compiling the code with the:
- `-W` or `--Wall` flag is useful, enabling a wide variety of usefull warnings
- `-Werror` treats critical warnings as errors, to make your code bulletproof once debugged.


```bash
%%bash
gcc ./hello_world.c -g -Werror -o hello_world
```

Now, we can finally execute our C code from Jupyter.


```bash
%%bash
./hello_world
```

    Hello World from C in Jupyter!

## Testing the C program: Python scripting

We can now take advantage of having a documentation system with shell and Python scripting capabilities to quickly develop input testing files to run multiple test cases against the C program. This effectively automates the unit testing of the C program.


```python

from itertools import permutations

inputs = permutations([3, '+', 43])
with open('permutations.in', 'w') as testfile:
  def str_from_perm(perm: list) -> str:
    s = str(perm)
    s = s.replace('(', ''); s = s.replace(')', ''); 
    s = s.replace("'", ''); s = s.replace(',', '')
    return s + '\n'
  [testfile.write(str_from_perm(perm)) for perm in inputs]
```


```python
%cat permutations.in
```

    3 + 43
    3 43 +
    + 3 43
    + 43 3
    43 3 +
    43 + 3



```python
%%file permutations.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

char *expected_tokens[] = {"3", "A", "+", "6", "43", NULL};

int arg_in_expected_tokens(char *arg, char **tokens){
  int found_token_chars = 0;
  for(char **it_token = tokens; *it_token != NULL; it_token++){
    if(!strcmp(arg, *it_token))
      return 1;
  }
  printf("Assertion Failed: %s not in [", arg);
  for(char **it_token = tokens; *it_token != NULL; ++it_token)
    printf("%s, ", *it_token);
  printf("]\n");
  return 0;
}

int main(int argc, char **argv) {
  assert(argc>1);
  for(int i = 1; i < argc; i++)
    assert(arg_in_expected_tokens(argv[i], expected_tokens));
  printf("Tests: Passed\n");
  exit(0);
}
```

    Overwriting permutations.c



```bash
%%bash
gcc ./permutations.c -g -Werror -o permutations
```

Finally, use bash shell scripting to input all inputs, line by line, to the C programm, so that each test is verified.


```bash
%%bash
echo "" > permutations.out
cat permutations.in | while read line
do
  ./permutations $line >> permutations.out
done
```


```python
%cat permutations.out
```

    
    Tests: Passed
    Tests: Passed
    Tests: Passed
    Tests: Passed
    Tests: Passed
    Tests: Passed


If you're developing in Linux, you can even use [Valgrind](http://valgrind.org/) to automatically detect many memory management and threading bugs, and profile your programs in detail.
