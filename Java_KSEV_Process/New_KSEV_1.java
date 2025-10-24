import java.io.*;
import java.util.*;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import java.nio.file.Files;
import java.nio.file.Paths;


public class New_KSEV_1 {

    public static void main(String[] args) throws IOException {
        String matrixFile = "C:/Users/fzkar/final_code_data/Java/Sample1661_transposed_matrices_by_4.txt";
        String bitstringFile = "C:/Users/fzkar/final_code_data/Java/new_best_str.txt";

        List<double[][]> group0 = new ArrayList<>();
        List<double[][]> group1 = new ArrayList<>();

        // Load matrices
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

        // Load bitstring
        String bitstring = new String(Files.readAllBytes(new File(bitstringFile).toPath())).trim();

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

        KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
        double[] pValues = new double[4];

        for (int i = 0; i < 4; i++) {
            double[] rowA = global0[i];
            double[] rowB = global1[i];
            pValues[i] = ks.kolmogorovSmirnovTest(rowA, rowB);
            System.out.printf("Row %d p-value: %.6f\n", i, pValues[i]);
        }

        double meanP = Arrays.stream(pValues).average().orElse(0);
        System.out.printf("\nFinal KSEV Score (mean p-value, lower = fitter): %.6f\n", meanP);
    }
// Optional: Add scaling or noise to eigenvalues to highlight small differences
	private double[] scaleOrAddNoise(double[] eigenvalues) {
		double scaleFactor = 1.00001; // Tiny scaling factor to introduce slight changes
		double noiseFactor = 0.00001; // Tiny noise factor
		for (int i = 0; i < eigenvalues.length; i++) {
			eigenvalues[i] = eigenvalues[i] * scaleFactor + (Math.random() * noiseFactor);
		}
		return eigenvalues;
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
    private static double[][] transpose(double[][] mat) {
        double[][] transposed = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    private static void normalizeRows(double[][] mat) {
        for (int i = 0; i < 4; i++) {
            double rowSum = 0;
            for (int j = 0; j < 4; j++) rowSum += mat[i][j];
            if (rowSum == 0) rowSum = 1;
            for (int j = 0; j < 4; j++) mat[i][j] /= rowSum;
        }
    }
    private static void printMatrix(double[][] matrix) {
    for (int i = 0; i < matrix.length; i++) {
        for (int j = 0; j < matrix[i].length; j++) {
            System.out.printf("%.4f\t", matrix[i][j]);
        }
        System.out.println();
    }
}

}