// ===== File 2: KSEVCalculatorFromNormalized.java =====

import java.io.*;
import java.util.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class KSEVCalculatorFromNormalized {

    public static void main(String[] args) throws IOException {
        String inputFile = "C:/Users/fzkar/final_code_data/Java/Normalized19Matrices.txt";
        String outputFile = "C:/Users/fzkar/final_code_data/Java/matrix_ksev19results.csv";

        List<double[][]> matrices = loadMatrices(inputFile);
        int n = matrices.size();

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("MatrixID_1,MatrixID_2,KSEV_Score");

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double[][] mat1 = matrices.get(i);
                    double[][] mat2 = matrices.get(j);

                    double score = computeKSEV(mat1, mat2, 13316); // Use fixed best seed

                    writer.printf("%d,%d,%.12f\n", i + 1, j + 1, score);
                }
            }
        }

        System.out.println("KSEV matrix computation completed.");
    }

    private static List<double[][]> loadMatrices(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        List<double[][]> matrices = new ArrayList<>();
        String line;
        double[][] matrix = new double[4][4];
        int row = 0;

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("Matrix ID:")) {
                continue;
            } else if (!line.isEmpty()) {
                String[] parts = line.split("\\s+");
                for (int i = 0; i < 4; i++) {
                    matrix[row][i] = Double.parseDouble(parts[i]);
                }
                row++;
                if (row == 4) {
                    matrices.add(matrix);
                    matrix = new double[4][4];
                    row = 0;
                }
            }
        }
        br.close();
        return matrices;
    }

    private static double computeKSEV(double[][] mat1, double[][] mat2, int seed) {
        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        double totalPValue = 0.0;
        Random rand = new Random(seed);

        for (int i = 0; i < 4; i++) {
            double[] row1 = mat1[i];
            double[] row2 = mat2[i];

            double[] eigenValues1 = calculateEigenvalues(row1);
            double[] eigenValues2 = calculateEigenvalues(row2);

            eigenValues1 = scaleOrAddNoise(eigenValues1, rand);
            eigenValues2 = scaleOrAddNoise(eigenValues2, rand);

            if (eigenValues1.length < 2 || eigenValues2.length < 2) {
                eigenValues1 = Arrays.copyOf(eigenValues1, 2);
                eigenValues1[1] = eigenValues1[0] + 1e-12;
                eigenValues2 = Arrays.copyOf(eigenValues2, 2);
                eigenValues2[1] = eigenValues2[0] + 1e-12;
            }

            double pValue = ksTest.kolmogorovSmirnovTest(eigenValues1, eigenValues2);
            totalPValue += pValue;
        }

        return totalPValue / 4.0;
    }

    private static double[] calculateEigenvalues(double[] row) {
        RealMatrix rowMatrix = new Array2DRowRealMatrix(row);
        RealMatrix covarianceMatrix = rowMatrix.multiply(rowMatrix.transpose());
        EigenDecomposition ed = new EigenDecomposition(covarianceMatrix);
        return ed.getRealEigenvalues();
    }

    private static double[] scaleOrAddNoise(double[] eigenvalues, Random rand) {
        double scaleFactor = 1.00001;
        double noiseFactor = 0.00001;
        for (int i = 0; i < eigenvalues.length; i++) {
            eigenvalues[i] = eigenvalues[i] * scaleFactor + (rand.nextDouble() * noiseFactor);
        }
        return eigenvalues;
    }
}
