import java.io.*;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class KSEVCalculator {

    public static void main(String[] args) throws IOException {
        String matrixFile = "C:/Users/fzkar/final_code_data/Java/Sample1661_transposed_matrices_by_4.txt";
        String bitstringFile = "C:/Users/fzkar/final_code_data/Java/the1315.txt";

        List<double[][]> group0 = new ArrayList<>();
        List<double[][]> group1 = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(matrixFile));
        String line;
        double[][] currentMatrix = new double[4][4];
        int row = 0;
        List<double[][]> allMatrices = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("Matrix ID:")) {
                if (row == 4) {
                    allMatrices.add(transpose(currentMatrix));
                    currentMatrix = new double[4][4];
                    row = 0;
                }
            } else if (!line.isEmpty()) {
                String[] parts = line.trim().split("\\s+");
                for (int i = 0; i < 4; i++) {
                    currentMatrix[row][i] = Double.parseDouble(parts[i]);
                }
                row++;
            }
        }
        if (row == 4) allMatrices.add(transpose(currentMatrix));
        br.close();

        String bitstring = new String(Files.readAllBytes(Paths.get(bitstringFile))).trim();

        for (int i = 0; i < bitstring.length(); i++) {
            if (bitstring.charAt(i) == '0') {
                group0.add(allMatrices.get(i));
            } else {
                group1.add(allMatrices.get(i));
            }
        }

        double[][] global0 = sumMatrices(group0);
        double[][] global1 = sumMatrices(group1);

        normalizeRows(global0);
        normalizeRows(global1);

        System.out.println("Normalized Global Matrix: Group 0");
        printMatrix(global0);
        System.out.println("Normalized Global Matrix: Group 1");
        printMatrix(global1);

        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        double totalPValue = 0.0;

        for (int i = 0; i < 4; i++) {
            double[] row1 = global0[i];
            double[] row2 = global1[i];

            double[] eigenValues1 = calculateEigenvalues(row1);
            double[] eigenValues2 = calculateEigenvalues(row2);

            eigenValues1 = scaleOrAddNoise(eigenValues1);
			eigenValues2 = scaleOrAddNoise(eigenValues2);


            double pValue = ksTest.kolmogorovSmirnovTest(eigenValues1, eigenValues2);
            System.out.printf("Row %d p-value: %.6f\n", i, pValue);
            totalPValue += pValue;
        }

        double ksevScore = totalPValue / 4.0;
        System.out.printf("\nFinal KSEV Score (mean p-value, lower = fitter): %.6f\n", ksevScore);
    }

    private static double[][] sumMatrices(List<double[][]> group) {
        double[][] sum = new double[4][4];
        for (double[][] mat : group) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    sum[i][j] += mat[i][j];
                }
            }
        }
        return sum;
    }

    private static void normalizeRows(double[][] mat) {
        for (int i = 0; i < 4; i++) {
            double rowSum = 0;
            for (int j = 0; j < 4; j++) rowSum += mat[i][j];
            if (rowSum == 0) rowSum = 1;
            for (int j = 0; j < 4; j++) mat[i][j] /= rowSum;
        }
    }

    private static double[] scaleOrAddNoise(double[] eigenvalues) {
        double scaleFactor = 1.00001;
        double noiseFactor = 0.00001;
        Random rand = new Random(1315); // Fixed seed for reproducibility
        for (int i = 0; i < eigenvalues.length; i++) {
            eigenvalues[i] = eigenvalues[i] * scaleFactor + (rand.nextDouble() * noiseFactor);
        }
        return eigenvalues;
    }
	

    private static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.printf("%.4f\t", matrix[i][j]);
            }
            System.out.println();
        }
    }

    private static double[][] transpose(double[][] mat) {
        double[][] transposed = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    private static double[] calculateEigenvalues(double[] row) {
        RealMatrix rowMatrix = new Array2DRowRealMatrix(row);
        RealMatrix covarianceMatrix = rowMatrix.multiply(rowMatrix.transpose());
        EigenDecomposition ed = new EigenDecomposition(covarianceMatrix);
        return ed.getRealEigenvalues();
    }
}
