import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;

public class Runner {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter your choice (0 for C program, 1 for Python program): ");
        int choice = scanner.nextInt();

        try {
            Process process;
            if (choice == 0) {
                process = new ProcessBuilder("./program").start();
            } else if (choice == 1) {
                process = new ProcessBuilder("python3", "script.py").start();
            } else {
                System.out.println("Invalid choice.");
                scanner.close();
                return;
            }

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            
            process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        scanner.close();
    }
}
