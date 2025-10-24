
import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;

public class KL_Fit_New_4x25 {

	private double bestKLAvgFitness = Double.MIN_VALUE;;
	private double bestEigenKLFitness = Double.MIN_VALUE;
	private double bestKSAVFitness = Double.MAX_VALUE;
	private double bestEigenKSFitness = Double.MAX_VALUE;

	private int bestKLAvgIteration = -1;
	private int bestEigenIteration = -1;
	private int bestKSAVIteration = -1;
	private int bestEigenKSIteration = -1;


	private double BestestFitness = 0;
	private int BestestIteration = 0;
	private int bestCountOnes = 0;
	
	private String bestBinaryString = ""; // To store the best binary string

	public static void main(String[] args) {
		KL_Fit_New_4x25 hc = new KL_Fit_New_4x25();
		Scanner scanner = new Scanner(System.in);

		// Ask user for fitness choice (between 0 and 3)
		System.out.println("Select fitness choice (0 for DoKLAV, 1 for DoKLEV, 2 for DoKSAV, 3 for DoKSEV):");
		int fitnessChoice = scanner.nextInt();

		long totalTime = 0;
		String outputFile = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HKAVEFITBestResults.csv";
		try (PrintWriter writer = new PrintWriter(new File(outputFile))) {
			writer.println("Best Fitness, Best Iteration, One Counts, Total Time Taken for the Total Run (ms), Best Binary String");


			for (int i = 1; i <= 1; i++) {
				
				long startTime = System.currentTimeMillis();

				// Reset best values for each iteration
				hc.bestKLAvgFitness = 0;
				hc.bestEigenKLFitness = 0;
				hc.bestKSAVFitness = Double.MAX_VALUE;
				hc.bestEigenKSFitness = Double.MAX_VALUE;
				hc.bestKLAvgIteration = -1;
				hc.bestEigenIteration = -1;
				hc.bestKSAVIteration = -1;
				hc.bestEigenKSIteration = -1;
				hc.bestCountOnes = 0;
				hc.BestestFitness = 0;
				hc.BestestIteration = 0;
				
				
				// int fitnessChoice = 0; // Example: 0 for DoKLAV, 1 for DoKLEV, etc.

				// kl.bestKLAvgBinaryString = "";
				// kl.bestEigenBinaryString = NULL;
				// Run your process
				int j = 0;
				int tot_iterations = 100000;
				// Z:\Data\Collaborations\Fawzia Z Kara-Isitt\FIT\FITNEW25_MAX
				String inputPath = "H:\\eclipseWorkspace\\KL\\data\\SwiftTest\\FIT\\Sample1661_transposed_matrices_by_4.txt";
				String klOutputPath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HK\\1HKKLVectorResults_"
						+ i + "_" + j + ".csv";
				String eigenOutputPath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HK\\1HKEigenVectorResults_"
						+ i + "_" + j + ".csv";
				String ksOutputPath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HK\\1HKKSAVResults_"
						+ i + "_" + j + ".csv";
				String eigenKSOutputPath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HK\\1HKEigenKSResults_"
						+ i + "_" + j + ".csv";
				String binaryStringPath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HKBinaryString_"
						+ i + "_" + j + ".csv";
				String flippedBitPath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\1HKBinaryFlippedBit_"
						+ i + "_" + j + ".csv";
				String pValueFilePath = "Z:\\Data\\Collaborations\\Fawzia Z Kara-Isitt\\FIT\\FITNEW25_MAX25_FINAL\\KSEV\\1HKKSEVPvalues_"
						+ i + "_" + j + ".csv";

				hc.processKLIterations(tot_iterations, j, inputPath, klOutputPath, eigenOutputPath, ksOutputPath,
						eigenKSOutputPath, pValueFilePath, binaryStringPath, flippedBitPath, fitnessChoice);

				long timeTaken = System.currentTimeMillis() - startTime;
				totalTime += timeTaken;

				// Capture the best values and their associated iterations
				// Assuming your `processKLIterations` method correctly updates the
				// `bestKLAvgFitness` and `bestEigenKLFitness`
				// during each iteration, you can directly use them here:

				// Write the current iteration's best results directly to the file
				//"Best Average Fitness KL, ,Best KS Avg Value,KS EV Value, EV Vectors, KL Iteration,Eigen Iteration,KSAV Iteration,KSEV Iteration, One Counts, Time Taken (ms)");
//0.180743157	0	1.7976931348623157E308	1.7976931348623157E308	996	-1	-1	-1	864	9161	

				writer.println(hc.BestestFitness + "," + hc.BestestIteration + "," + hc.bestCountOnes + "," + timeTaken + "," + hc.bestBinaryString);
	}
		} catch (FileNotFoundException e) {
			System.out.println(e.getMessage());
		}

	}

	public void processKLIterations(int iterations, int it, String inputPath, String klOutputPath,
			String eigenOutputPath, String ksOutputPath, String eigenKSOutputPath, String pValueFilePath,
			String binaryStringPath, String flippedBitPath, int fitnessChoice) {

		long startTime = System.currentTimeMillis(); // Record start time
		long totalItTime = 0;
		try (PrintWriter klWriter = (fitnessChoice == 0) ? new PrintWriter(new FileWriter(klOutputPath, true)) : null;
				PrintWriter eigenWriter = (fitnessChoice == 1) ? new PrintWriter(new FileWriter(eigenOutputPath, true))
						: null;
				PrintWriter KSWriter = (fitnessChoice == 2) ? new PrintWriter(new FileWriter(ksOutputPath, true))
						: null;
				PrintWriter eigenKSWriter = (fitnessChoice == 3)
						? new PrintWriter(new FileWriter(eigenKSOutputPath, true))
						: null;
				PrintWriter pValueEVKSWriter = (fitnessChoice == 3)
						? new PrintWriter(new FileWriter(pValueFilePath, true))
						: null;
				PrintWriter binaryWriter = (fitnessChoice == 0 || fitnessChoice == 1 || fitnessChoice == 2
						|| fitnessChoice == 3) ? new PrintWriter(new FileWriter(binaryStringPath, true)) : null;
				PrintWriter flippedBitWriter = (fitnessChoice == 0 || fitnessChoice == 1 || fitnessChoice == 2
						|| fitnessChoice == 3) ? new PrintWriter(new FileWriter(flippedBitPath, true)) : null) {

			Map<Integer, double[][]> matricesMap = readMatrices(inputPath);
			List<Pair<Integer, double[][]>> matricesWithIds = new ArrayList<>();

// Collect matrices with IDs
			for (Map.Entry<Integer, double[][]> entry : matricesMap.entrySet()) {
				matricesWithIds.add(new Pair<>(entry.getKey(), entry.getValue()));
			}

			Random random = new Random();
			bestBinaryString = binaryVectorToString(generateRandomBinaryVector(matricesWithIds.size(), random)); // Initialize
																													// best
																													// binary
																													// string
			List<Integer> binaryVector = stringToBinaryList(bestBinaryString); // Convert bestBinaryString to a list for
																				// initial processing

			List<Pair<Integer, double[][]>> group0 = new ArrayList<>();
			List<Pair<Integer, double[][]>> group1 = new ArrayList<>();
			distributeGroups(binaryVector, matricesWithIds, group0, group1);

			double[][] sumMatrix0 = normalizeMatrix(sumMatrices(group0));
			double[][] sumMatrix1 = normalizeMatrix(sumMatrices(group1));

			switch (fitnessChoice) {
			case 0:
				bestKLAvgFitness = DoKLAV(sumMatrix0, sumMatrix1);
				break;
			case 1:
				bestEigenKLFitness = DoKLEV(sumMatrix0, sumMatrix1);
				break;
			case 2:
				bestKSAVFitness = DoKSAV(sumMatrix0, sumMatrix1, it);
				break;
			case 3:
				bestEigenKSFitness = DoKSEV(sumMatrix0, sumMatrix1, pValueFilePath);
				break;
			default:
				throw new IllegalArgumentException("Invalid fitness choice: " + fitnessChoice);
			}

// Process iterations (mutations)
			for (int majoriteration = 0; majoriteration < iterations; majoriteration++) {
				long ItstartTime = System.currentTimeMillis();
				

// Mutate the best binary vector
				List<Integer> mutatedBinaryVector = stringToBinaryList(bestBinaryString); // Convert bestBinaryString to
																							// a list
				int flippedBitIndex = mutateAndGetFlippedBitIndex(mutatedBinaryVector, random); // Flip a bit in the
																								// list
				String mutatedBinaryString = binaryVectorToString(mutatedBinaryVector); // Convert back to string

				int CountOnes = countOnesInString(mutatedBinaryString);
				//System.out.println("Mutated Binary String: " + mutatedBinaryString);
				//System.out.println("Indices with value '1': " + CountOnes);
				//System.out.flush();
				
// Save the mutated binary string and flipped bit if required
				if (binaryWriter != null) {
					binaryWriter.println(mutatedBinaryString);
				}

				if (flippedBitWriter != null) {
					flippedBitWriter
							.println("Iteration " + majoriteration + ", Flipped Bit Position: " + flippedBitIndex);
				}

// Continue with the existing logic
				List<Pair<Integer, double[][]>> group0new = new ArrayList<>();
				List<Pair<Integer, double[][]>> group1new = new ArrayList<>();
				distributeGroups(mutatedBinaryVector, matricesWithIds, group0new, group1new);

				double[][] sumMatrix0new = normalizeMatrix(sumMatrices(group0new));
				double[][] sumMatrix1new = normalizeMatrix(sumMatrices(group1new));

// Choose the fitness calculation based on the input parameter
				double fitnessValueNew = 0.0;
				switch (fitnessChoice) {
				case 0:
					fitnessValueNew = DoKLAV(sumMatrix0new, sumMatrix1new);
					if (Math.abs(fitnessValueNew) >= Math.abs(bestKLAvgFitness)) {
						bestKLAvgFitness = fitnessValueNew;
						bestKLAvgIteration = majoriteration;
						bestBinaryString = mutatedBinaryString; // Update best binary string
					}
					if (klWriter != null) {
						saveKLResults(klWriter, majoriteration, bestKLAvgFitness, fitnessValueNew, CountOnes,totalItTime);
					}
					BestestFitness = bestKLAvgFitness;
					BestestIteration = majoriteration;
					break;
				case 1:
					fitnessValueNew = DoKLEV(sumMatrix0new, sumMatrix1new);
					if (Math.abs(fitnessValueNew) >= Math.abs(bestEigenKLFitness)) {
						bestEigenKLFitness = fitnessValueNew;
						bestEigenIteration = majoriteration;
						bestBinaryString = mutatedBinaryString; // Update best binary string
					}
					if (eigenWriter != null) {
						saveEigenResults(eigenWriter, majoriteration, bestEigenKLFitness, fitnessValueNew, CountOnes, totalItTime);
					}
					BestestFitness = bestEigenKLFitness;
					BestestIteration = majoriteration;
					break;
				case 2:
					fitnessValueNew = DoKSAV(sumMatrix0new, sumMatrix1new, majoriteration);
					if (Math.abs(fitnessValueNew) <= Math.abs(bestKSAVFitness)) {
						bestKSAVFitness = fitnessValueNew;
						bestKSAVIteration = majoriteration;
						bestBinaryString = mutatedBinaryString; // Update best binary string
					}
					if (KSWriter != null) {
						saveKSAVResults(KSWriter, majoriteration, bestKSAVFitness, fitnessValueNew, CountOnes,totalItTime);
					}
					BestestFitness = bestKSAVFitness;
					BestestIteration = majoriteration;
					break;
				case 3:
					fitnessValueNew = DoKSEV(sumMatrix0new, sumMatrix1new, pValueFilePath);
					if (Math.abs(fitnessValueNew) <= Math.abs(bestEigenKSFitness)) {
						bestEigenKSFitness = fitnessValueNew;
						bestEigenKSIteration = majoriteration;
						bestBinaryString = mutatedBinaryString; // Update best binary string
					}
					if (eigenKSWriter != null) {
						saveEigenKSResults(eigenKSWriter, majoriteration, bestEigenKSFitness, fitnessValueNew,CountOnes,
								totalItTime);
					}
					BestestFitness = bestEigenKSFitness;
					BestestIteration = majoriteration;
					break;
				default:
					throw new IllegalArgumentException("Invalid fitness choice: " + fitnessChoice);
				}

// Capture time for this iteration
				long ItTime = System.currentTimeMillis();
				totalItTime = ItTime - ItstartTime;

// Update the best binary vector for the next iteration
				binaryVector = stringToBinaryList(bestBinaryString);
				//System.out.println("Iteration " + majoriteration + " completed. Fitness Value: " + fitnessValueNew
				//		+ ", Time taken: " + totalItTime + "ms");
				bestCountOnes = CountOnes;
				
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void processKLIterationsOld2(int iterations, int it, String inputPath, String klOutputPath,
			String eigenOutputPath, String ksOutputPath, String eigenKSOutputPath, String pValueFilePath,
			String binaryStringPath, String flippedBitPath, int fitnessChoice) {

		long startTime = System.currentTimeMillis(); // Record start time

// Open writers for selected fitness choice files
		try (PrintWriter klWriter = (fitnessChoice == 0) ? new PrintWriter(new FileWriter(klOutputPath, true)) : null;
				PrintWriter eigenWriter = (fitnessChoice == 1) ? new PrintWriter(new FileWriter(eigenOutputPath, true))
						: null;
				PrintWriter KSWriter = (fitnessChoice == 2) ? new PrintWriter(new FileWriter(ksOutputPath, true))
						: null;
				PrintWriter eigenKSWriter = (fitnessChoice == 3)
						? new PrintWriter(new FileWriter(eigenKSOutputPath, true))
						: null;
				PrintWriter pValueEVKSWriter = (fitnessChoice == 3)
						? new PrintWriter(new FileWriter(pValueFilePath, true))
						: null;
				PrintWriter binaryWriter = (fitnessChoice == 0 || fitnessChoice == 1 || fitnessChoice == 2
						|| fitnessChoice == 3) ? new PrintWriter(new FileWriter(binaryStringPath, true)) : null;
				PrintWriter flippedBitWriter = (fitnessChoice == 0 || fitnessChoice == 1 || fitnessChoice == 2
						|| fitnessChoice == 3) ? new PrintWriter(new FileWriter(flippedBitPath, true)) : null) {

			Map<Integer, double[][]> matricesMap = readMatrices(inputPath);
			List<Pair<Integer, double[][]>> matricesWithIds = new ArrayList<>();

// Collect matrices with IDs
			for (Map.Entry<Integer, double[][]> entry : matricesMap.entrySet()) {
				matricesWithIds.add(new Pair<>(entry.getKey(), entry.getValue()));
			}
			double fitnessValueNew = 0.0;
			Random random = new Random();
			List<Integer> binaryVector = generateRandomBinaryVector(matricesWithIds.size(), random); // Initial binary
																										// vector

			List<Pair<Integer, double[][]>> group0 = new ArrayList<>();
			List<Pair<Integer, double[][]>> group1 = new ArrayList<>();
			distributeGroups(binaryVector, matricesWithIds, group0, group1);

			double[][] sumMatrix0 = normalizeMatrix(sumMatrices(group0));
			double[][] sumMatrix1 = normalizeMatrix(sumMatrices(group1));

			// double[][] sumMatrix0new;
			// double[][] sumMatrix1new;

			switch (fitnessChoice) {
			case 0:
				bestKLAvgFitness = DoKLAV(sumMatrix0, sumMatrix1);
				break;
			case 1:
				bestEigenKLFitness = DoKLEV(sumMatrix0, sumMatrix1);
				break;
			case 2:
				bestKSAVFitness = DoKSAV(sumMatrix0, sumMatrix1, it);
				break;
			case 3:
				bestEigenKSFitness = DoKSEV(sumMatrix0, sumMatrix1, pValueFilePath);
				break;
			default:
				throw new IllegalArgumentException("Invalid fitness choice: " + fitnessChoice);
			}

// Process 100 iterations (mutations)
			for (int majoriteration = 0; majoriteration < iterations; majoriteration++) {
				long ItstartTime = System.currentTimeMillis();
				long totalItTime = 0;

// Mutate the binary vector and record the flipped bit position
				List<Integer> binaryVectorManipulated = new ArrayList<>(binaryVector); // Copy of the original vector
				int flippedBitIndex = mutateAndGetFlippedBitIndex(binaryVectorManipulated, random); // Get the flipped
																									// bit index

// Convert mutated vector to string
				String mutatedBinaryString = binaryVectorToString(binaryVectorManipulated);
				int CountOnes = countOnesInString(mutatedBinaryString);
				//System.out.println("Mutated Binary String: " + mutatedBinaryString);
				//System.out.println("Indices with value '1': " + CountOnes);
				//System.out.flush();
// Log the mutated binary string and flipped bit if the choice is 0, 1, 2, or 3
				if (binaryWriter != null) {
					binaryWriter.println(mutatedBinaryString);
				}

				if (flippedBitWriter != null) {
					flippedBitWriter
							.println("Iteration " + majoriteration + ", Flipped Bit Position: " + flippedBitIndex);
				}

// Continue with the existing logic
				List<Pair<Integer, double[][]>> group0new = new ArrayList<>();
				List<Pair<Integer, double[][]>> group1new = new ArrayList<>();
				distributeGroups(binaryVectorManipulated, matricesWithIds, group0new, group1new);

				double[][] sumMatrix0new = normalizeMatrix(sumMatrices(group0new));
				double[][] sumMatrix1new = normalizeMatrix(sumMatrices(group1new));

// Choose the fitness calculation based on the input parameter

				switch (fitnessChoice) {
				case 0:
					fitnessValueNew = DoKLAV(sumMatrix0new, sumMatrix1new);
					if (Math.abs(fitnessValueNew) >= Math.abs(bestKLAvgFitness)) {
						bestKLAvgFitness = fitnessValueNew;
						bestKLAvgIteration = majoriteration;
						bestBinaryString = mutatedBinaryString;
					}
					if (klWriter != null) {
						saveKLResults(klWriter, majoriteration, bestKLAvgFitness, fitnessValueNew, CountOnes, totalItTime);
					}
					break;
				case 1:
					fitnessValueNew = DoKLEV(sumMatrix0new, sumMatrix1new);
					if (Math.abs(fitnessValueNew) >= Math.abs(bestEigenKLFitness)) {
						bestEigenKLFitness = fitnessValueNew;
						bestEigenIteration = majoriteration;
					}
					if (eigenWriter != null) {
						saveEigenResults(eigenWriter, majoriteration, bestEigenKLFitness, fitnessValueNew,CountOnes, totalItTime);
					}
					break;
				case 2:
					fitnessValueNew = DoKSAV(sumMatrix0new, sumMatrix1new, majoriteration);
					if (Math.abs(fitnessValueNew) <= Math.abs(bestKSAVFitness)) {
						bestKSAVFitness = fitnessValueNew;
						bestKSAVIteration = majoriteration;
					}
					if (KSWriter != null) {
						saveKSAVResults(KSWriter, majoriteration, bestKSAVFitness, fitnessValueNew,CountOnes, totalItTime);
					}
					break;
				case 3:
					fitnessValueNew = DoKSEV(sumMatrix0, sumMatrix1, pValueFilePath);
					if (Math.abs(fitnessValueNew) <= Math.abs(bestEigenKSFitness)) {
						bestEigenKSFitness = fitnessValueNew;
						bestEigenKSIteration = majoriteration;
					}
					if (eigenKSWriter != null) {
						saveEigenKSResults(eigenKSWriter, majoriteration, bestEigenKSFitness, fitnessValueNew,CountOnes,
								totalItTime);
					}
					break;
				default:
					throw new IllegalArgumentException("Invalid fitness choice: " + fitnessChoice);
				}

// Capture time for this iteration
				long ItTime = System.currentTimeMillis();
				totalItTime = ItTime - ItstartTime;

// Update the original binary vector with the manipulated one for the next iteration
				binaryVector = new ArrayList<>(binaryVectorManipulated);

// Show progress of iteration
				//System.out.println("Iteration " + majoriteration + " completed. Fitness Value: " + fitnessValueNew
				//		+ ", Time taken: " + totalItTime + "ms");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private List<Integer> stringToBinaryList(String binaryString) {
		List<Integer> binaryList = new ArrayList<>();
		for (char c : binaryString.toCharArray()) {
			if (c == '0' || c == '1') {
				binaryList.add(c - '0'); // Convert '0'/'1' to 0/1
			} else {
				throw new IllegalArgumentException("Invalid character in binary string: " + c);
			}
		}
		return binaryList;
	}

	// Mutate the binary string and return the index of the flipped bit
	private int mutateAndGetFlippedBitIndex(List<Integer> binaryVector, Random random) {
		int indexToFlip = random.nextInt(binaryVector.size());
		binaryVector.set(indexToFlip, binaryVector.get(indexToFlip) == 0 ? 1 : 0);
		return indexToFlip;
	}

	private int countOnesInString(String binaryString) {
		int count = 0;

		for (int i = 0; i < binaryString.length(); i++) {
			if (binaryString.charAt(i) == '1') {
				count++;
			}
		}

		return count;
	}

	private double DoKLAV(double[][] m0, double[][] m1) {
		double kl = 0.0;
		for (int i = 0; i < m0.length; ++i) {
			double x = KL(m0[i], m1[i]);
			kl += x;
		}
		kl /= (double) (m0.length);
		return kl;
	}

	private double DoKLEV(double[][] m0, double[][] m1) {
		// Transpose matrices before computing eigenvectors
		double[] p0 = GetConv(transposeMatrix(m0));
		double[] p1 = GetConv(transposeMatrix(m1));

		double kl = KL(p0, p1);
		return kl;
	}

	private double DoKSAV(double[][] m0, double[][] m1, int iterationks) {
		// Initialize the Kolmogorov-Smirnov Test
		KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();

		// Transpose the input matrices to facilitate row-by-row comparison
		double[][] xm0 = transposeMatrix(m0);
		double[][] xm1 = transposeMatrix(m1);

		int xrows = xm0.length; // Number of rows after transposition
		int xcols = xm0[0].length; // Number of columns after transposition

		double xtotalPValue = 0; // Accumulate p-values
		double[][] xksResults = new double[xrows][xcols]; // To store KS test results for each row

		// Perform KS test row by row between corresponding rows of xm0 and xm1
		for (int i = 0; i < xrows; i++) {
			// Extract the i-th row from both transposed matrices
			double[] xsample1 = xm0[i];
			double[] xsample2 = xm1[i];

			// Perform the Kolmogorov-Smirnov test for the two samples
			double xpValue = ksTest.kolmogorovSmirnovTest(xsample1, xsample2);

			// Store the result in the results matrix
			xksResults[i][0] = xpValue; // Assuming we're interested in the p-value for each row

			// Accumulate the p-value for fitness calculation
			xtotalPValue += xpValue;
		}

		// Calculate the average p-value (fitness value) by dividing the total by the
		// number of rows
		double xaveragePValue = xtotalPValue / xrows;

		// Optionally, save the KS results for debugging or further analysis
		// saveKSVectors(xksResults, "Z:\\Data\\Collaborations\\Fawzia Z
		// Kara-Isitt\\FIT\\FITNEW25_MAX0\\xks_vectors1" + iterationks + ".csv");

		return xaveragePValue; // Return the average p-value as the fitness result
	}

	public double DoKSEV(double[][] matrix1, double[][] matrix2, String pValueFile) {
		KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
		double totalPValue = 0;
		double[] pValues = new double[matrix1.length];

		// Log matrix contents for debugging before applying the KS test
		// System.out.println("Matrix1 before KS test:");
		// printMatrix(matrix1);
		// System.out.println("Matrix2 before KS test:");
		// printMatrix(matrix2);

		try (PrintWriter writer = new PrintWriter(pValueFile)) {
			for (int i = 0; i < matrix1.length; i++) {
				// Get the rows (vectors) from each matrix
				double[] row1 = matrix1[i];
				double[] row2 = matrix2[i];

				// Calculate eigenvalues for each row
				double[] eigenValues1 = calculateEigenvalues(row1);
				double[] eigenValues2 = calculateEigenvalues(row2);

				// Log eigenvalues for each row for debugging
				// System.out.println("Eigenvalues for row " + i + " of Matrix1: " +
				// Arrays.toString(eigenValues1));
				// System.out.println("Eigenvalues for row " + i + " of Matrix2: " +
				// Arrays.toString(eigenValues2));

				// Apply scaling or slight noise to ensure detection of small variations
				// (optional)
				eigenValues1 = scaleOrAddNoise(eigenValues1);
				eigenValues2 = scaleOrAddNoise(eigenValues2);

				// Perform the KS test on the eigenvalues
				double pValue = ksTest.kolmogorovSmirnovTest(eigenValues1, eigenValues2);
				pValues[i] = pValue; // Store pValue in the array

				// Write the p-value to the file
				writer.println(pValue);

				// Accumulate the p-value for fitness calculation
				totalPValue += pValue;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Return the average p-value as the fitness value
		return totalPValue / matrix1.length;
	}

	private double KL(double[] p, double[] q) {
	    double kl = 0.0;
	    double epsilon = 1e-10; // Small value to prevent division by zero and log of zero
	    for (int i = 0; i < p.length; ++i) {
	        try {
	            double adjustedP = p[i] + epsilon;
	            double adjustedQ = q[i] + epsilon;
	            double x = adjustedP * Math.log(adjustedP / adjustedQ);
	            if (Double.isFinite(x)) {
	                kl += x;
	            }
	        } catch (Exception e) {
	            kl += 0.0;
	        }
	    }
	    return kl;
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

	/**
	 * Calculate eigenvalues for a given row vector (treated as a part of a matrix).
	 */
	private double[] calculateEigenvalues(double[] row) {
		// Create a 1x4 matrix from the row
		RealMatrix rowMatrix = new Array2DRowRealMatrix(row);

		// To calculate eigenvalues, we'll use a square matrix (covariance-like)
		RealMatrix covarianceMatrix = rowMatrix.multiply(rowMatrix.transpose());

		// Calculate eigenvalues using Apache Commons Math
		EigenDecomposition eigenDecomposition = new EigenDecomposition(covarianceMatrix);
		return eigenDecomposition.getRealEigenvalues();
	}

	private double[] calculateEigenvaluesOld(double[] row) {
		// Check if the row is valid
		if (row == null || row.length == 0) {
			throw new IllegalArgumentException("Row vector is null or empty.");
		}

		// Create a 1xN matrix from the row
		RealMatrix rowMatrix = new Array2DRowRealMatrix(row);

		// To calculate eigenvalues, we'll use a square matrix (covariance-like)
		RealMatrix covarianceMatrix = rowMatrix.multiply(rowMatrix.transpose());

		try {
			// Calculate eigenvalues using Apache Commons Math
			EigenDecomposition eigenDecomposition = new EigenDecomposition(covarianceMatrix);
			return eigenDecomposition.getRealEigenvalues();
		} catch (Exception e) {
			System.err.println("Error during eigenvalue calculation: " + e.getMessage());
			return new double[row.length]; // Return an array of zeros as a fallback
		}
	}

	private double[] GetConv(double[][] m) {
		RealMatrix matrix = MatrixUtils.createRealMatrix(m);
		// RealMatrix matrix = createRealMatrix(m);
		EigenDecomposition eigenDecomp = new EigenDecomposition(matrix);

		// Get the eigenvector associated with the largest eigenvalue
		double[] largestEigenvector = eigenDecomp.getEigenvector(0).toArray();

		// Normalize the eigenvector so that the sum of its elements equals 1
		double sum = 0.0;
		for (double v : largestEigenvector) {
			sum += Math.abs(v); // Use absolute values to avoid negative sum issues
		}
		for (int i = 0; i < largestEigenvector.length; ++i) {
			largestEigenvector[i] /= sum;
		}

		// System.out.println("Normalized Eigenvector: ");
		// for (double v : largestEigenvector) {
		// System.out.print(v + " ");
		// }
		System.out.println();

		return largestEigenvector;
	}

	private double[] GetConvOld(double[][] m) {
		if (m == null || m.length == 0 || m[0].length == 0) {
			throw new IllegalArgumentException("Matrix is null or empty.");
		}

		// Ensure all rows have the same length
		for (int i = 1; i < m.length; i++) {
			if (m[i].length != m[0].length) {
				throw new IllegalArgumentException("Inconsistent row lengths in the matrix.");
			}
		}

		RealMatrix matrix;
		try {
			matrix = MatrixUtils.createRealMatrix(m);
		} catch (Exception e) {
			System.err.println("Error creating real matrix: " + e.getMessage());
			return new double[m.length]; // Return an array of zeros as a fallback
		}

		try {
			EigenDecomposition eigenDecomp = new EigenDecomposition(matrix);

			// Get the eigenvector associated with the largest eigenvalue
			double[] largestEigenvector = eigenDecomp.getEigenvector(0).toArray();

			// Normalize the eigenvector so that the sum of its elements equals 1
			double sum = Arrays.stream(largestEigenvector).map(Math::abs).sum(); // Use absolute values to avoid
																					// negative sum issues

			if (sum == 0) {
				System.err.println("Sum of eigenvector elements is zero. Returning zero vector.");
				return new double[largestEigenvector.length]; // Return zero vector if sum is zero
			}

			for (int i = 0; i < largestEigenvector.length; ++i) {
				largestEigenvector[i] /= sum;
			}

			return largestEigenvector;
		} catch (Exception e) {
			System.err.println("Error during eigenvalue decomposition: " + e.getMessage());
			return new double[m.length]; // Return an array of zeros as a fallback
		}
	}

	private double[][] transposeMatrix(double[][] matrix) {
		int rows = matrix.length;
		int cols = matrix[0].length;
		double[][] transposed = new double[cols][rows];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				transposed[j][i] = matrix[i][j];
			}
		}

		return transposed;
	}

	private void saveKLResults(PrintWriter writer, int iteration, double avgKL, double avgKLnew, int OneCounts, long totalItTime) {
		DecimalFormat df = new DecimalFormat("0.000000000000");
		// writer.println("Iteration: " + iteration + ", Average KL Divergence: " +
		// df.format(Math.abs(avgKL))+ ", New KL Divergence: " +
		// df.format(Math.abs(avgKLnew))+ ", Iteration Time (ms): " + totalItTime);
		writer.println(iteration + ",  " + df.format(Math.abs(avgKL)) + ", " + df.format(Math.abs(avgKLnew)) + ",  "+OneCounts+",  "
				+ totalItTime);

	}

	private void saveEigenResults(PrintWriter writer, int iteration, double eigenKL, double eigenKLnew,int OneCounts,
			long totalItTime) {
		DecimalFormat df = new DecimalFormat("0.000000000000");
		// writer.println("Iteration: " + iteration + ", Eigenvector KL Divergence: " +
		// df.format(Math.abs(eigenKL)));
		writer.println(iteration + ",  " + df.format(Math.abs(eigenKL)) + ", " + df.format(Math.abs(eigenKLnew)) + ",  "+OneCounts+",  "
				+ totalItTime);
	}

	private void saveKSAVResults(PrintWriter writer, int iteration, double KSKL, double KSKLnew, int OneCounts, long totalItTime) {
		DecimalFormat df = new DecimalFormat("0.000000000000");
		// writer.println("Iteration: " + iteration + ", KolmogorovSmirnov KL
		// Divergence: " + df.format(Math.abs(eigenKL)));
		writer.println(iteration + ",  " + df.format(Math.abs(KSKL)) + ", " + df.format(Math.abs(KSKLnew)) + ",  "
				+ totalItTime);

	}

	private void saveEigenKSResults(PrintWriter writer, int iteration, double eigenKL, double eigenKLnew,int OneCounts,
			long totalItTime) {
		DecimalFormat df = new DecimalFormat("0.000000000000");
		// writer.println("Iteration: " + iteration + ", Eigenvector KL Divergence: " +
		// df.format(Math.abs(eigenKL)));
		writer.println(iteration + ",  " + df.format(Math.abs(eigenKL)) + ", " + df.format(Math.abs(eigenKLnew)) + ",  "+OneCounts+",  "
				+ totalItTime);
	}
	
	//0.180743157	0	1.7976931348623157E308	1.7976931348623157E308	996	-1	-1	-1	864	9161	


	private void saveBestResultsAndTime(PrintWriter writer, String fitnessType, double bestFitness, int bestIteration,int OneCounts,
			long totalTime) {
		DecimalFormat df = new DecimalFormat("0.000000000000");
		writer.println(
				"Best " + fitnessType + ": " + df.format(Math.abs(BestestFitness)) + ", Iteration: " + BestestIteration+ ", One Counts: " +OneCounts+",  ");
		writer.println("Total Processing Time (ms): " + totalTime);
	}

	private void saveKSVectors(double[][] ksResults, String fileName) {
		try (PrintWriter writer = new PrintWriter(new FileWriter(fileName))) {
			for (int i = 0; i < ksResults.length; i++) {
				for (int j = 0; j < ksResults[i].length; j++) {
					writer.print(ksResults[i][j]);
					if (j < ksResults[i].length - 1) {
						writer.print(",");
					}
				}
				writer.println();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Utility function to save matrices to a CSV file for logging
	private void saveMatrixToFile(double[][] matrix, String filename) {
		try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
			for (double[] row : matrix) {
				writer.println(Arrays.toString(row).replaceAll("[\\[\\]]", "")); // Remove square brackets
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void saveBinaryString(PrintWriter writer, int iteration, String binaryString) {
		writer.println("Iteration: " + iteration + ", Binary String: " + binaryString);
	}

	private Map<Integer, double[][]> readMatrices(String filePath) throws IOException {
		Map<Integer, double[][]> matrices = new HashMap<>();
		try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
			String line;
			double[][] matrix = new double[4][4];
			int row = 0;
			Integer currentId = null;

			while ((line = reader.readLine()) != null) {
				line = line.trim();
				if (line.startsWith("Matrix ID:")) {
					if (currentId != null && row == 4) {
						matrices.put(currentId, matrix);
					}
					currentId = Integer.parseInt(line.substring(11).trim());
					matrix = new double[4][4];
					row = 0;
				} else if (!line.isEmpty()) {
					String[] values = line.split("\\s+");
					if (values.length != 4) {
						throw new IllegalArgumentException(
								"Expected 4 values per row, found " + values.length + ": Matrix ID: " + currentId);
					}
					for (int i = 0; i < values.length; i++) {
						matrix[row][i] = Double.parseDouble(values[i]);
					}
					row++;
				}
			}
			if (currentId != null && row == 4) {
				matrices.put(currentId, matrix);
			}
		}
		return matrices;
	}

	private List<Integer> generateRandomBinaryVector(int size, Random random) {
		return random.ints(size, 0, 2).boxed().collect(Collectors.toList());
	}

	// Method to mutate a single bit in the binary string and save the position
	// Method to mutate a single bit in the binary string and save the position
	private List<Integer> mutateBinaryString(List<Integer> binaryVector) {
		// Convert the binary list to a character array for easy manipulation
		Random random = new Random();

		// Randomly select a bit index to flip
		int indexToFlip = random.nextInt(binaryVector.size());

		// Flip the selected bit (from '0' to '1' or from '1' to '0')
		binaryVector.set(indexToFlip, binaryVector.get(indexToFlip) == 0 ? 1 : 0);

		// Return the mutated binary vector
		return binaryVector;
	}

	private double[][] sumMatrices(List<Pair<Integer, double[][]>> group) {
		double[][] sumMatrix = new double[4][4];
		for (Pair<Integer, double[][]> pair : group) {
			double[][] matrix = pair.getSecond();
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					sumMatrix[i][j] += matrix[i][j];
				}
			}
		}
		return sumMatrix;
	}

	private double[][] normalizeMatrix(double[][] matrix) {
	    double[][] normalizedMatrix = new double[4][4];
	    double totalSum = 0.0;

	    // Calculate the total sum of all elements in the matrix
	    for (int i = 0; i < 4; i++) {
	        totalSum += Arrays.stream(matrix[i]).sum();
	    }

	    // Normalize the matrix
	    for (int i = 0; i < 4; i++) {
	        for (int j = 0; j < 4; j++) {
	            normalizedMatrix[i][j] = totalSum == 0 ? 0 : matrix[i][j] / totalSum;
	        }
	    }

	    return normalizedMatrix;
	}


	private String binaryVectorToString(List<Integer> binaryVector) {
		StringBuilder sb = new StringBuilder();
		for (Integer bit : binaryVector) {
			sb.append(bit);
		}
		return sb.toString();
	}

	private void distributeGroups(List<Integer> binaryVector, List<Pair<Integer, double[][]>> matricesWithIds,
			List<Pair<Integer, double[][]>> group0, List<Pair<Integer, double[][]>> group1) {
		if (binaryVector.size() != matricesWithIds.size()) {
			throw new IllegalArgumentException("The size of the binary vector must match the number of matrices.");
		}

		for (int i = 0; i < binaryVector.size(); i++) {
			Pair<Integer, double[][]> matrixEntry = matricesWithIds.get(i);
			if (binaryVector.get(i) == 0) {
				group0.add(matrixEntry);
			} else {
				group1.add(matrixEntry);
			}
		}
	}

	private static class Pair<T, U> {
		private final T first;
		private final U second;

		public Pair(T first, U second) {
			this.first = first;
			this.second = second;
		}

		public T getFirst() {
			return first;
		}

		public U getSecond() {
			return second;
		}
	}
}
