import java.io.*;
import java.util.*;
import java.io.*;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class KSEVCalculator_25K {

    public static void main(String[] args) throws IOException {
        String matrixFile = "C:/Users/fzkar/final_code_data/Java/Sample1661_transposed_matrices_by_4.txt";
        String allBitstringsFile = "C:/Users/fzkar/final_code_data/Java/the1315.txt";
        String outputFile = "C:/Users/fzkar/final_code_data/Java/best_ksev_result1315.txt";
        
        List<double[][]> allMatrices = loadMatrices(matrixFile);
        List<String> lines = Files.readAllLines(Paths.get(allBitstringsFile));

        double bestScore = Double.MAX_VALUE;
        String bestBitstring = "";
        int bestIndex = -1;
        int currentIndex = 0;
        int validCount = 0;

        for (String line : lines) {
            String[] parts = line.split(",");
            String bitstring = parts[parts.length - 1].trim().replaceAll("[^01]", "");

            if (bitstring.length() != allMatrices.size()) {
                System.err.println("Skipping line " + currentIndex + " due to invalid bitstring length: " + bitstring.length());
                currentIndex++;
                continue;
            }

            if (currentIndex % 100 == 0) {
                System.out.println("Processing bitstring index: " + currentIndex);
            }

            List<double[][]> group0 = new ArrayList<>();
            List<double[][]> group1 = new ArrayList<>();

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
                //double pValue = ksTest.kolmogorovSmirnovTest(row1, row2);

                totalPValue += pValue;
            }

            double ksevScore = totalPValue / 4.0;
            validCount++;

            if (ksevScore < bestScore) {
                bestScore = ksevScore;
                bestBitstring = bitstring;
                bestIndex = currentIndex;
            }

            currentIndex++;
        }

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("Best KSEV Score: " + bestScore);
            writer.println("Bitstring Index: " + bestIndex);
            writer.println("Bitstring producing best score: " + bestBitstring);
            writer.println("Total Valid Bitstrings Processed: " + validCount);
        }

        System.out.println("Best KSEV Score: " + bestScore);
        System.out.println("Bitstring Index: " + bestIndex);
        System.out.println("Bitstring producing best score: " + bestBitstring);
        System.out.println("Total Valid Bitstrings Processed: " + validCount);
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
                    matrices.add((currentMatrix));
                   // matrices.add(transpose(currentMatrix));

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
        //if (row == 4) matrices.add(transpose(currentMatrix));
        if (row == 4) matrices.add((currentMatrix));
        br.close();
        return matrices;
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

    private static double[][] transpose(double[][] mat) {
        double[][] transposed = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }
	// Optional: Add scaling or noise to eigenvalues to highlight small differences
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
}
