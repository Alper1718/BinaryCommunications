#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef _WIN32
    #include <windows.h>
    #define PLAY_SOUND(bit) PlaySound(bit ? "sound1.wav" : "sound0.wav", NULL, SND_FILENAME | SND_ASYNC)
#elif __APPLE__
    #define PLAY_SOUND(bit) system(bit ? "afplay sound1.wav" : "afplay sound0.wav")
#else
    #define PLAY_SOUND(bit) system(bit ? "aplay sound1.wav" : "aplay sound0.wav")
#endif

void play_binary_audio(int value, int is_int) {
    PLAY_SOUND(is_int);

    for (int i = 7; i >= 0; i--) {
        PLAY_SOUND((value >> i) & 1);
    }
}

int main() {
    char input[100];

    printf("Enter a mix of characters and numbers (space-separated): ");
    fgets(input, sizeof(input), stdin);

    char *token = strtok(input, " \n");
    while (token != NULL) {
        int is_int = 1;

        for (int i = 0; token[i] != '\0'; i++) {
            if (!isdigit(token[i])) {
                is_int = 0;
                break;
            }
        }

        if (is_int) {
            int num = atoi(token);
            if (num > 255 || num < 0) {
                printf("Number out of 8-bit range: %d\n", num);
            } else {
                play_binary_audio(num, 1);
            }
        } else {
            for (int i = 0; token[i] != '\0'; i++) {
                play_binary_audio(token[i], 0);
            }
        }

        token = strtok(NULL, " \n");
    }

    return 0;
}
