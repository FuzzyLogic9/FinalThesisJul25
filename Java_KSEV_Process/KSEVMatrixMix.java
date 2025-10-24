import java.io.*;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class KSEVMatrixMix {

    public static void main(String[] args) throws IOException {
        String matrixFile = "C:/Users/fzkar/final_code_data/Java/Sample19_transposed_matrices.txt";
        String outputFile = "C:/Users/fzkar/final_code_data/Java/matrix_ksev_19x19_results.csv";

        List<double[][]> matrices = loadMatrices(matrixFile);
        int n = matrices.size();

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("MatrixID_1,MatrixID_2,KSEV_Score");

            for (int i = 0; i < 22; i++) {
                for (int j = 0; j < 22; j++) {
                    double[][] mat1 = matrices.get(i);
                    double[][] mat2 = matrices.get(j);

                    double[][] norm1 = deepCopy(mat1);
                    double[][] norm2 = deepCopy(mat2);


                    normalizeRows(norm1);
                    normalizeRows(norm2);

                    int bestSeed = 13316;

                    double ksevScore = computeKSEV(norm1, norm2, bestSeed);

                    writer.printf("%d,%d,%.12f%n", i + 1, j + 1, ksevScore);
                }
            }
        }

        System.out.println("22x22 KSEV matrix computation completed.");
    }

    private static List<double[][]> loadMatrices(String matrixFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        String line;
        double[][] currentMatrix = new double[4][4];
        int row = 0;
        List<double[][]> matrices = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("Matrix ID:")) {
                if (row == 4) {
                    matrices.add(currentMatrix);
                    currentMatrix = new double[4][4];
                    row = 0;
                }
            } else if (!line.isEmpty()) {
                String[] parts = line.split("\\s+");
                for (int i = 0; i < 4; i++) {
                    currentMatrix[row][i] = Double.parseDouble(parts[i]);
                }
                row++;
            }
        }
        if (row == 4) matrices.add(currentMatrix);
        br.close();
        return matrices;
    }

    private static void normalizeRows(double[][] mat) {
        for (int i = 0; i < 4; i++) {
            double rowSum = 0;
            for (int j = 0; j < 4; j++) rowSum += mat[i][j];
            if (rowSum == 0) rowSum = 1;
            for (int j = 0; j < 4; j++) mat[i][j] /= rowSum;
        }
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
    // Pad both arrays with a small duplicate value to meet KS test requirement
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

    private static double[] scaleOrAddNoise(double[] eigenvalues) {
        double scaleFactor = 1.00001;
        double noiseFactor = 0.00001;
        for (int i = 0; i < eigenvalues.length; i++) {
            eigenvalues[i] = eigenvalues[i] * scaleFactor + (Math.random() * noiseFactor);
        }
        return eigenvalues;
    }

    private static double[] calculateEigenvalues(double[] row) {
        RealMatrix rowMatrix = new Array2DRowRealMatrix(row);
        RealMatrix covarianceMatrix = rowMatrix.multiply(rowMatrix.transpose());
        EigenDecomposition ed = new EigenDecomposition(covarianceMatrix);
        return ed.getRealEigenvalues();
    }

    private static double[][] deepCopy(double[][] original) {
        double[][] copy = new double[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            System.arraycopy(original[i], 0, copy[i], 0, original[i].length);
        }
        return copy;
    }
}
