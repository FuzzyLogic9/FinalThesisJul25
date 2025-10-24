// ===== File 1: NormalizeAndSave.java =====

import java.io.*;
import java.util.*;

public class NormalizeAndSaveMat {
    public static void main(String[] args) throws IOException {
        String inputFile = "C:/Users/fzkar/final_code_data/Java/Sample4_19.txt";
        String outputFile = "C:/Users/fzkar/final_code_data/Java/Normalized4_19Matrices.txt";

        List<double[][]> matrices = loadAndNormalize(inputFile);

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            int id = 1;
            for (double[][] matrix : matrices) {
                writer.println("Matrix ID: " + id++);
                for (double[] row : matrix) {
                    for (int j = 0; j < row.length; j++) {
                        writer.printf("%.6f%s", row[j], j < row.length - 1 ? " " : "\n");
                    }
                }
                writer.println();
            }
        }

        System.out.println("Normalization complete. Output written to: " + outputFile);
    }

    private static List<double[][]> loadAndNormalize(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        List<double[][]> matrices = new ArrayList<>();
        String line;
        double[][] current = new double[4][4];
        int row = 0;

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("Matrix ID:")) {
                if (row == 4) {
                    normalize(current);
                    matrices.add(current);
                    current = new double[4][4];
                    row = 0;
                }
            } else if (!line.isEmpty()) {
                String[] parts = line.split("\\s+");
                for (int i = 0; i < 4; i++) {
                    current[row][i] = Double.parseDouble(parts[i]);
                }
                row++;
            }
        }
        if (row == 4) {
            normalize(current);
            matrices.add(current);
        }
        br.close();
        return matrices;
    }

    private static void normalize(double[][] mat) {
        for (int i = 0; i < 4; i++) {
            double sum = 0;
            for (int j = 0; j < 4; j++) sum += mat[i][j];
            if (sum == 0) sum = 1;
            for (int j = 0; j < 4; j++) mat[i][j] /= sum;
        }
    }
}
