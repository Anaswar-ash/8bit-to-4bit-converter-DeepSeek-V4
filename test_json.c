#include <stdio.h>
#include <stdlib.h>
#include "tok.h"

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    Tok T = {0};
    printf("Calling tok_load...\\n");
    tok_load(&T, "v4_int4/tokenizer.json");
    printf("tok_load finished successfully! n_ids = %d\\n", T.n_ids);
    return 0;
}
