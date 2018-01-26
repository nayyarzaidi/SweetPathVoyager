package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import datastructure.Parameter;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import weka.core.converters.ArffLoader.ArffReader;


public class SUtils {

	public static int minNumThreads = 4000;
	public static int displayPerfAfterInstances = 1000;
	public static String perfOutput = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-+*`~!@#$%^&_|:;'?";
	public static int m_Limit = 1;
	public static int seed = 1;


	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static double MEsti(double freq1, double freq2, double numValues) {
		double m_MEsti = 1.0;
		double mEsti = (freq1 + m_MEsti / numValues) / (freq2 + m_MEsti);
		return mEsti;
	}

	public static double MEsti(double freq1, double freq2) {
		double mEsti = freq1 / freq2;
		return mEsti;
	}

	public static void boundAndNormalizeInLogDomain(double[] logs,
			double maxDifference) {
		boundDifferences(logs, maxDifference);
		double logSum = sumInLogDomain(logs);
		for (int i = 0; i < logs.length; i++)
			logs[i] -= logSum;
	}

	public static void boundDifferences(double[] logs, double maxDifference) {
		double maxLog = logs[0];
		for (int i = 1; i < logs.length; i++) {
			if (maxLog < logs[i]) {
				maxLog = logs[i];
			}
		}
		for (int i = 0; i < logs.length; i++) {
			logs[i] = logs[i] - maxLog;
			if (logs[i] < -maxDifference) {
				logs[i] = -maxDifference;
			}
		}
	}

	public static void normalizeInLogDomain(double[] logs) {
		double logSum = sumInLogDomain(logs);
		for (int i = 0; i < logs.length; i++)
			logs[i] -= logSum;
	}

	public static double sumInLogDomain(double[] logs) {
		// first find max log value
		double maxLog = logs[0];
		int idxMax = 0;
		for (int i = 1; i < logs.length; i++) {
			if (maxLog < logs[i]) {
				maxLog = logs[i];
				idxMax = i;
			}
		}
		// now calculate sum of exponent of differences
		double sum = 0;
		for (int i = 0; i < logs.length; i++) {
			if (i == idxMax) {
				sum++;
			} else {
				sum += Math.exp(logs[i] - maxLog);
			}
		}
		// and return log of sum
		return maxLog + Math.log(sum);
	}

	public static void exp(double[] logs) {
		for (int c = 0; c < logs.length; c++) {
			logs[c] = Math.exp(logs[c]);
		}
	}

	public static double[] exp2(double[] logs) {
		double[] a = new double[logs.length];
		for (int c = 0; c < logs.length; c++) {
			a[c] = Math.exp(logs[c]);
		}
		return a;
	}

	public static void log(double[] logs) {
		for (int c = 0; c < logs.length; c++) {
			logs[c] = Math.log(logs[c]);
		}
	}	

	public static int[] sort(double[] mi) {		
		int[] sortedPositions = Utils.sort(mi);
		int n = mi.length;
		int[] order = new int[n];
		for (int i = 0; i < n; i++) {
			order[i] = sortedPositions[(n-1) - i];
		}
		return order;
	}

	public static int combination(int N, int k) {
		int n = 0;
		int num = factorial(N);
		int denum1 = factorial(N - k);
		int denum2 = factorial(k);
		n = (num) / (denum1 * denum2);
		return n;
	}

	public static int factorial(int a) {
		int facta = 1;		
		for (int i = a; i > 0; i--) {
			facta *= i;
		}		
		return facta;
	}

	public static int NC2(int a) {
		int count = 0;
		for (int att1 = 1; att1 < a; att1++) {			
			for (int att2 = 0; att2 < att1; att2++) {
				count++;	
			}
		}
		return count;		
	}

	public static int NC3(int a) {
		int count = 0;
		for (int att1 = 2; att1 < a; att1++) {			
			for (int att2 = 1; att2 < att1; att2++) {				
				for (int att3 = 0; att3 < att2; att3++) {
					count++;
				}
			}
		}
		return count;		
	}

	public static int NC4(int a) {
		int count = 0;

		for (int att1 = 3; att1 < a; att1++) {			
			for (int att2 = 2; att2 < att1; att2++) {				
				for (int att3 = 1; att3 < att2; att3++) {
					for (int att4 = 0; att4 < att3; att4++) {
						count++;
					}
				}
			}
		}
		return count;
	}

	public static int NC5(int a) {
		int count = 0;

		for (int att1 = 4; att1 < a; att1++) {			
			for (int att2 = 3; att2 < att1; att2++) {				
				for (int att3 = 2; att3 < att2; att3++) {
					for (int att4 = 1; att4 < att3; att4++) {
						for (int att5 = 0; att5 < att4; att5++) {
							count++;
						}
					}
				}
			}
		}
		return count;
	}

	public final static void randomize(int[] index, int n) {

		Random random = new Random(System.currentTimeMillis());

		for (int i = 0; i < index.length; i++) {
			int k = random.nextInt(n);
			index[i] = k;
		}

	}

	public final static void randomize(int[] index) {
		Random random = new Random(System.currentTimeMillis());
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}

	public final void randomize(int[] index, Random random) {
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}

	public static void shuffleArray(int[] ar) {

		Random rnd = new Random();
		for (int i = ar.length - 1; i > 0; i--)
		{
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}

	public static double maxAbsValueInAnArray(double[] array) {
		int index = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (Math.abs(array[i]) > max) {
				max = Math.abs(array[i]);
				index = i;				
			}
		}		
		return Math.abs(array[index]);
	}

	public static int maxLocationInAnArray(double[] array) {
		int index = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				index = i;				
			}
		}		
		return index;
	}

	public static int minLocationInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] < min) {
				min = array[i];
				index = i;				
			}
		}		
		return index;
	}

	public static int findMaxValueLocationInNDMatrix(double[][] results, int dim) {
		double[] tempVector = new double[results.length];

		for (int i = 0; i < results.length; i++) {
			tempVector[i] = results[i][dim];
		}

		int index = minLocationInAnArray(tempVector);

		return index;
	}

	public static double MI(long[][] contingencyMatrix) {
		int n = 0;
		int nrows = contingencyMatrix.length;
		int ncols = contingencyMatrix[0].length;

		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += contingencyMatrix[r][c];
				colsSum[c] += contingencyMatrix[r][c];
				n += contingencyMatrix[r][c];
			}				
		}

		double MI = 0;

		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (contingencyMatrix[r][c] > 0) {
							double a = contingencyMatrix[r][c] / ( rowsSum[r]/(double)n * colsSum[c] ) ;
							MI += contingencyMatrix[r][c]/(double)n * Math.log(a / Math.log(2));
						}
					}
				}
			}
		}
		return MI;
	}

	public static double gsquare(long[][]observed){
		long n = 0;
		int nrows = observed.length;
		int ncols = observed[0].length;

		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += observed[r][c];
				colsSum[c] += observed[r][c];
				n += observed[r][c];
			}				
		}

		double gs = 0.0;
		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (observed[r][c] > 0) {
							double exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n) ;
							gs+= 1.0*observed[r][c]*FastMath.log(observed[r][c]/exp);
						}
					}
				}
			}
		}
		gs*=2.0;
		return gs;
	}

	public static double chisquare(long[][]observed){
		long n = 0;
		int nrows = observed.length;
		int ncols = observed[0].length;

		int[] rowsSum = new int[nrows];
		int[] colsSum = new int[ncols];

		for (int r = 0; r < nrows; r++) {
			for (int c = 0; c < ncols; c++) {
				rowsSum[r] += observed[r][c];
				colsSum[c] += observed[r][c];
				n += observed[r][c];
			}				
		}

		double chi = 0.0;
		for (int r = 0; r < nrows; r++) {
			if (rowsSum[r] != 0) {
				for (int c = 0; c < ncols; c++) {
					if (colsSum[c] != 0) {
						if (observed[r][c] > 0) {
							double exp = (1.0*rowsSum[r]/n) * (1.0*colsSum[c]/n) ;
							double diff = observed[r][c]-exp;
							chi+= diff*diff/exp;
						}
					}
				}
			}
		}
		return chi;
	}

	public static boolean inSet(int[] array, int element) {
		boolean present = false;

		if (array == null) {
			return false;
		}

		for (int i = 0; i < array.length; i++) {
			if (array[i] == element) {
				present = true;
				break;
			}
		}
		return present;
	}

	public static boolean linkExist(int[][] m_Parents, int[] m_Order, int i, int j) {
		boolean present = false;

		if (inSet(m_Parents[i], m_Order[j]))
			present = true;		

		if (inSet(m_Parents[j], m_Order[i]))
			present = true;

		return present;
	}

	public static int[] CheckForPerfectness(int[] m_TempParents, int[][] m_Parents, int[] m_Order) {
		int[] parents = null;		
		parents = new int[m_TempParents.length];

		int j = 0;
		for (int j1 = 0; j1 < m_TempParents.length; j1++) {
			for (int j2 = j1 + 1; j2 < m_TempParents.length; j2++) {

				if (SUtils.linkExist(m_Parents, m_Order, m_TempParents[j1], m_TempParents[j2])) {
					parents[j] = m_TempParents[j1];
					parents[j+1] = m_TempParents[j2];
					j+=2;
				}

			}
		}	

		return null;
	}

	public static void labelsToProbs(int[] labels, double[] probs) {
		int[] count = new int[probs.length];
		for (int i = 0; i < labels.length; i++) {
			count[labels[i]]++;
		}
		int maxCount = Integer.MIN_VALUE;
		int labelIndex = -1;
		for (int i = 0; i < count.length; i++) {
			if (count[i] > maxCount) {
				maxCount = count[i];
				labelIndex = i;
			}
		}

		probs[labelIndex] = 1.0;
	}

	public static boolean inArray(int val, int[] arr) {

		boolean flag = false;

		if (arr == null) {
			return flag;
		}

		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == val) {
				flag = true;
			}
		}
		return flag;
	}

	public static double minAbsValueInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (Math.abs(array[i]) < min) {
				min = Math.abs(array[i]);
				index = i;				
			}
		}		
		return Math.abs(array[index]);
	}

	public static double minNonZeroValueInAnArray(double[] array) {
		int index = 0;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] != 0 && array[i] < min) {
				min = Math.abs(array[i]);
				index = i;				
			}
		}		
		return array[index];
	}

	public static Instances generateBaggedData(Instances instances) {
		int N = instances.numInstances();
		Random random = new Random(System.currentTimeMillis());

		Instances data = new Instances(instances, 0);

		for (int i = 0; i < N; i++) {
			int index = random.nextInt(N);
			data.add(instances.instance(index));
		}

		return data;
	}

	public static void normalize(double[] ds) {
		double sum = 0.0;
		for (int i = 0; i < ds.length; i++) {
			sum += ds[i];
		}
		if (sum > 0) {
			for (int i = 0; i < ds.length; i++) {
				ds[i] /= sum;
			}		
		}
	}

	public static boolean monotonic(double[] results) {

		double[] diff = new double[results.length - 1];

		for (int i = 0; i < results.length - 2; i++) {
			diff[i] = results[i + 1] - results[i];
		}

		int numSadlePoints = 0;

		for (int i = 0; i < diff.length - 2; i++) {
			if (!sameSign(diff[i + 1], diff[i]))  {
				numSadlePoints++;
			}
		}

		if (numSadlePoints > 1) 
			return false;
		else
			return true;

	}

	private static boolean sameSign(double a, double b) {
		return ((a<0) == (b<0)); 
	}

	/*
	 * -------------------------------------------------------------------------------------
	 * Cross-validation functions
	 * -------------------------------------------------------------------------------------
	 */

	public static ArrayList<Integer> getTrainTestIndices(int N) {

		int Nvalidation = 0;

		if (N / 10 >= 10000) {
			Nvalidation = 10000;
		} else {
			Nvalidation = N / 10;
		}

		System.out.println("Creating Validation (CV) file of size: " + Nvalidation);

		MersenneTwister rg = new MersenneTwister();

		ArrayList<Integer> indexList = new ArrayList<>();

		int nvalid = 0;
		while (nvalid < Nvalidation) {
			int index = rg.nextInt(N);
			if (!indexList.contains(index)) {
				indexList.add(index);
				nvalid++;
			}
		}

		return indexList;
	}

	public static Instances[] getTrainTestInstances(Instances cvInstances) {

		Instances[] instancesList = new Instances[2];

		Instances temp = Globals.getStructure();

		instancesList[0] = new Instances(temp);
		instancesList[1] = new Instances(temp);

		int N = cvInstances.numInstances();

		for (int i = 0; i < N; i++) {
			Instance row = cvInstances.instance(i);

			if (i % 10 == 0) {
				instancesList[1].add(row);
			} else {
				instancesList[0].add(row);
			}

		}

		System.out.println("-- Train Test files created for cross-validating step size -- Train = " + instancesList[0].numInstances() + ", and Test = " + instancesList[1].numInstances());

		return instancesList;

	}

	public static Instances[] getTrainTestInstances(File sourceFile, ArrayList<Integer> indexList, int BUFFER_SIZE) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances[] instancesList = new Instances[2];

		Instances instancesTrain = new Instances(structure);
		Instances instancesTest = new Instances(structure);

		int nvalidation = indexList.size();

		int i = 0;
		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			if (indexList.contains(i)) {
				if (nvalidation % 5 == 0) {
					instancesTest.add(row);
				} else {
					instancesTrain.add(row);
				}
				nvalidation++;
			}
			i++;
		}

		instancesList[0] = instancesTrain;
		instancesList[1] = instancesTrain;


		System.out.println("-- Train Test files created for cross-validating step size -- Train = " + instancesTrain.numInstances() + ", and Test = " + instancesTest.numInstances());

		return instancesList;
	}

	public static BitSet getStratifiedIndices(File sourceFile, int BUFFER_SIZE, int ARFF_BUFFER_SIZE, int S) throws FileNotFoundException, IOException {

		BitSet res = new BitSet();

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), ARFF_BUFFER_SIZE);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();

		int[] classCount = new int[nc];
		int[] numToBeSelected = new int[nc];
		int[] numSelected = new int[nc];
		double[] selectionProb = new double[nc];

		Instance current;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();
			classCount[x_C]++;
		}

		for (int c = 0; c < nc; c++) {
			if (classCount[c] < 50) {
				numToBeSelected[c] = classCount[c]/2;
				selectionProb[c] = 0.5;
			} else { 
				numToBeSelected[c] = (classCount[c]/100) * S;
				selectionProb[c] = (double) S / 100;
			}
		}

		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), ARFF_BUFFER_SIZE);
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();

			if (Math.random() < selectionProb[x_C]) {
				res.set(lineNo);
				numSelected[x_C]++;
			}

			lineNo++;
		}

		System.out.println("-------------------------------------------------------------------");
		System.out.println("Class Counts = " + Arrays.toString(classCount));
		System.out.println("Num to be selected = " + Arrays.toString(numToBeSelected));
		System.out.println("Actually selected = " + Arrays.toString(numSelected));
		System.out.println("-------------------------------------------------------------------");

		return res;
	}

	public static BitSet getStratifiedIndices() throws FileNotFoundException, IOException {

		BitSet res = new BitSet();
		int S = Globals.getHoldoutsetPrecentage();

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();
		int n = structure.numAttributes() - 1;

		int[] classCount = new int[nc];
		int[] numToBeSelected = new int[nc];
		int[] numSelected = new int[nc];
		double[] selectionProb = new double[nc];

		Instance current;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();
			classCount[x_C]++;
		}

		for (int c = 0; c < nc; c++) {
			if (classCount[c] < 50) {
				numToBeSelected[c] = classCount[c]/2;
				selectionProb[c] = 0.5;
			} else { 
				numToBeSelected[c] = (classCount[c]/100) * S;
				selectionProb[c] = (double) S / 100;
			}
		}

		reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();

			if (Math.random() < selectionProb[x_C]) {
				res.set(lineNo);
				numSelected[x_C]++;
			}

			lineNo++;
		}

		System.out.println("-------------------------------------------------------------------");
		System.out.println("Class Counts = " + Arrays.toString(classCount));
		System.out.println("Num to be selected = " + Arrays.toString(numToBeSelected));
		System.out.println("Actually selected = " + Arrays.toString(numSelected));
		System.out.println("-------------------------------------------------------------------");

		Globals.setNumberInstances(lineNo - res.cardinality());
		Globals.setNumInstancesKnown(true);

		return res;
	}

	public static BitSet getStratifiedIndices(int S, String data) throws FileNotFoundException, IOException {

		BitSet res = new BitSet();

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(data), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();

		int[] classCount = new int[nc];
		int[] numToBeSelected = new int[nc];
		int[] numSelected = new int[nc];
		double[] selectionProb = new double[nc];

		Instance current;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();
			classCount[x_C]++;
		}

		for (int c = 0; c < nc; c++) {
			if (classCount[c] < 50) {
				numToBeSelected[c] = classCount[c]/2;
				selectionProb[c] = 0.5;
			} else { 
				numToBeSelected[c] = (classCount[c]/100) * S;
				selectionProb[c] = (double) S / 100;
			}
		}

		reader = new ArffReader(new BufferedReader(new FileReader(data), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			int x_C = (int) current.classValue();

			if (Math.random() < selectionProb[x_C]) {
				res.set(lineNo);
				numSelected[x_C]++;
			}

			lineNo++;
		}

		if (Globals.isVerbose()) {
			System.out.println("-------------------------------------------------------------------");
			System.out.println("Class Counts = " + Arrays.toString(classCount));
			System.out.println("Num to be selected = " + Arrays.toString(numToBeSelected));
			System.out.println("Actually selected = " + Arrays.toString(numSelected));
			System.out.println("-------------------------------------------------------------------");
		}

		return res;
	}

	public static Instances getTrainTestInstances(BitSet res) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances CVInstances = new Instances(structure);

		int i = 0;
		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			if (res.get(i)) {
				CVInstances.add(row);
			}
			i++;
		}

		System.out.println("-- CVInstances file created (in memory) -- Size = " + CVInstances.numInstances());

		return CVInstances;
	}

	public static Instances getTrainTestInstances(File file) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(file), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances CVInstances = new Instances(structure);

		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			CVInstances.add(row);
		}

		System.out.println("-- CVInstances file created (in memory) -- Size = " + CVInstances.numInstances());

		return CVInstances;
	}

	public static Instances getTrainTestInstances(File sourceFile, BitSet res, int BUFFER_SIZE, int ARFF_BUFFER_SIZE) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), ARFF_BUFFER_SIZE);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instances CVInstances = new Instances(structure);

		int i = 0;
		Instance row;
		while ((row = reader.readInstance(structure)) != null)  {
			if (res.get(i)) {
				CVInstances.add(row);
			}
			i++;
		}

		System.out.println("-- CVInstances file created (in memory) -- Size = " + CVInstances.numInstances());

		return CVInstances;
	}

	public static File discretizeData(File sourceFile, Instances CVInstances, int BUFFER_SIZE, int ARFF_BUFFER_SIZE, int type, int numBins) throws Exception {

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Instances m_DiscreteInstances = null;

		if (type == 1) {

			/* MDL based discretization */

			System.out.println("Starting MDL Discretization");

			weka.filters.supervised.attribute.Discretize m_Disc = null;

			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(CVInstances);

			m_DiscreteInstances = weka.filters.Filter.useFilter(CVInstances, m_Disc);

			out.deleteOnExit();

			ArffSaver fileSaver = new ArffSaver();
			fileSaver.setFile(out);
			fileSaver.setRetrieval(Saver.INCREMENTAL);
			fileSaver.setStructure(m_DiscreteInstances);

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			int i = 0;
			while ((row = reader.readInstance(structure)) != null)  {
				m_Disc.input(row);
				row = m_Disc.output();

				fileSaver.writeIncremental(row);
				i++;

			}
			fileSaver.writeIncremental(null);

		} else if (type == 2) {

			/* Equal Frequency based discretization */

			System.out.println("Starting Equal Frequency Discretization with " + numBins + " bins.");

			weka.filters.unsupervised.attribute.Discretize m_Disc = null;

			m_Disc = new weka.filters.unsupervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(CVInstances);
			m_Disc.setBins(numBins);

			m_DiscreteInstances = weka.filters.Filter.useFilter(CVInstances, m_Disc);

			out.deleteOnExit();

			ArffSaver fileSaver = new ArffSaver();
			fileSaver.setFile(out);
			fileSaver.setRetrieval(Saver.INCREMENTAL);
			fileSaver.setStructure(m_DiscreteInstances);

			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);

			Instance row;
			int i = 0;
			while ((row = reader.readInstance(structure)) != null)  {
				m_Disc.input(row);
				row = m_Disc.output();

				fileSaver.writeIncremental(row);
				i++;
			}
			fileSaver.writeIncremental(null);

		}

		return out;
	}

	public static File normalizeData(File sourceFile, Instances CVInstances, int BUFFER_SIZE, int ARFF_BUFFER_SIZE) throws Exception {

		System.out.println("Starting Normalization");

		Instances m_NormalizedInstances = null;

		weka.filters.unsupervised.attribute.Normalize m_Norm = null;

		m_Norm = new weka.filters.unsupervised.attribute.Normalize();
		m_Norm.setInputFormat(CVInstances);

		m_NormalizedInstances = weka.filters.Filter.useFilter(CVInstances, m_Norm);

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(m_NormalizedInstances);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), ARFF_BUFFER_SIZE);
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance row;
		int i = 0;
		while ((row = reader.readInstance(structure)) != null)  {
			m_Norm.input(row);
			row = m_Norm.output();

			fileSaver.writeIncremental(row);
			i++;
		}
		fileSaver.writeIncremental(null);

		return out;
	}

	public static BitSet getTest0Indexes(int N, MersenneTwister rg)  {
		BitSet res = new BitSet();
		int nLines = 0;
		for (int i = 0; i < N; i++) {
			if (rg.nextBoolean()) {
				res.set(nLines);
			}
			nLines++;
		}

		int expectedNLines = (nLines % 2 == 0) ? nLines / 2 : nLines / 2 + 1;
		int actualNLines = res.cardinality();

		if (actualNLines < expectedNLines) {
			while (actualNLines < expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (res.get(chosen));
				res.set(chosen);
				actualNLines++;
			}
		} else if (actualNLines > expectedNLines) {
			while (actualNLines > expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (!res.get(chosen));
				res.clear(chosen);
				actualNLines--;
			}
		}
		return res;
	}

	public static File createTrainTmpFile(Instances structure, BitSet trainIndexes) throws IOException {

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()),Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());

		Instance current;
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			if (trainIndexes.get(lineNo)) {
				fileSaver.writeIncremental(current);
			}
			lineNo++;
		}
		fileSaver.writeIncremental(null);
		return out;
	}

	public static Instances setStructure() throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		int n = structure.numAttributes() - 1;
		int nc = structure.numClasses();

		boolean[] isNumericTrue = new boolean[n];
		int[] paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			if (structure.attribute(u).isNominal()) { 
				isNumericTrue[u] = false;
				paramsPerAtt[u] = structure.attribute(u).numValues();
			} else if (structure.attribute(u).isNumeric()) {
				isNumericTrue[u] = true;
				paramsPerAtt[u] = 1;
			}
		}

		Globals.setNumAttributes(n);
		Globals.setNumClasses(nc);

		Globals.setParamsPerAtt(paramsPerAtt);
		Globals.setIsNumericTrue(isNumericTrue);

		Globals.setStructure(structure);

		return structure;
	}

	public static double determineNumData() throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		Instance current;
		double lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			lineNo++;
		}

		return lineNo;
	}


	//	public static Instances getStructure() throws FileNotFoundException, IOException {
	//		
	//		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
	//		Instances structure = reader.getStructure();
	//		structure.setClassIndex(structure.numAttributes() - 1);
	//		
	//		return structure;
	//	}

	public static BitSet combineIndexes(BitSet[] indexes, int fold) {
		BitSet foldIndexes = new BitSet();

		for (int i = 0; i < Globals.getNumFolds(); i++) {
			if (i != fold) {
				foldIndexes.or(indexes[i]);
			}
		}

		return foldIndexes;
	}

	public static void getIndexes(BitSet[] indexes) {
		int numFolds = Globals.getNumFolds();
		int N = (int) Globals.getNumberInstances();

		int expectedNumberInEachBin = N/numFolds;
		int leftOvers = N % numFolds;

		for (int i = 0; i < N; i++) {
			int bin = getBin(numFolds);
			indexes[bin].set(i);
		}

		Queue<Integer> movers = new LinkedList<Integer>();

		for (int i = 0; i < numFolds; i++) {
			int diff = indexes[i].cardinality() - expectedNumberInEachBin;

			if (diff > 0) {
				while (indexes[i].cardinality() != expectedNumberInEachBin) {
					int j = (int) (Math.random() * N);
					if (indexes[i].get(j)) {
						movers.add(j);
						indexes[i].clear(j);
					}
				}
			}

		}

		for (int i = 0; i < numFolds; i++) {
			int diff = indexes[i].cardinality() - expectedNumberInEachBin;

			if (diff < 0) {
				while (indexes[i].cardinality() != expectedNumberInEachBin) {
					if (movers.isEmpty()) {
						break;
					}
					int j = movers.remove();
					indexes[i].set(j);
				}

			}
		}


	}

	private static int getBin(int numFolds) {
		double[] randProbs = new double[numFolds];
		for (int i = 0; i < numFolds; i++) {
			randProbs[i] = Math.random();
		}
		int bin = SUtils.maxLocationInAnArray(randProbs);
		return bin;
	}

	public static double[] getResults(double[] probs, int x_C, int nc) {

		double[] results = new double[4];
		double rmse = 0;
		double loss = 0;
		double neglogloss = 0;

		int pred = -1;
		double bestProb = Double.MIN_VALUE;
		for (int y = 0; y < nc; y++) {
			if (!Double.isNaN(probs[y])) {
				if (probs[y] > bestProb) {
					pred = y;
					bestProb = probs[y];
				}
				rmse += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
			} else {
				//System.out.println("Woopsy Daisy");
				//System.err.println("probs[ " + y + "] is NaN! oh no!");

				//System.exit(-1);
			}
		}

		if (pred != x_C) {
			loss = 1;
		}

		neglogloss = - Math.log(probs[x_C]);

		results[0] = rmse;
		results[1] = loss;
		results[2] = neglogloss;
		results[3] = pred;

		return results;
	}

	public static File randomizeTrainingFile() throws IOException {

		int N = (int) Globals.getNumberInstances();

		int[] randIndices = getPermuatation(N);

		Map cache = new HashMap(); 

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("(randomizeTrainingFile): Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(Globals.getSOURCEFILE()), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		fileSaver.setStructure(structure);

		Instance row;
		int i = 0;
		int lineNo = 0;
		while ((row = reader.readInstance(structure)) != null)  {

			cache.put(lineNo, row);

			int lookingFor = randIndices[i];

			if (lineNo == lookingFor) {

				for (int j = i; j < randIndices.length && randIndices[j] <= lineNo; j++) {
					int extract = randIndices[j];

					Instance InstanceC = (Instance) cache.get(extract);
					fileSaver.writeIncremental(InstanceC);

					cache.remove(extract);
					i++;
				}

			}

			lineNo++;
		}
		fileSaver.writeIncremental(null);

		if (Globals.isVerbose()) {
			System.out.println("New Source File after Randomization is: " + out.getAbsolutePath());
		}

		return out;

	}

	private static int[] getPermuatation(int n) {
		int[] indices = new int[n];
		BitSet bindices = new BitSet();

		Random rand = new Random();
		int randomNum = 0;

		for (int i = 0; i < n; i++) {

			do {
				randomNum = rand.nextInt(n);
			} while (bindices.get(randomNum));

			bindices.set(randomNum);
			indices[i] = randomNum;
		}

		return indices;
	}


	public static void deleteSourceFile() {
		File out = Globals.getSOURCEFILE();
		out.delete();
	}

	public static boolean isPrime(int n) {

		if (n < 2) {
			return false;
		} else if (n == 2) {
			return true;
		} else if (n % 2 == 0) {
			return false;
		}else {
			int sqrtN = (int) Math.sqrt(n);
			for (int i = 3; i <= sqrtN + 1; i+=2) {
				if (n%i == 0) {
					System.out.println(n + "is divisible by: " + i);
					return false;
				}
			}
		}

		return true;
	}

	public static int getPreviousPrime(int n) {
		int pn = 1;

		for (int i = n-1; i >= 2; i--) {
			if (isPrime(i)) {
				return i;
			}
		}

		return pn;
	}

	public static double computeMutualInformation(int u1, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		double m = 0;
		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {

			int xcount = 0;
			for (int y = 0; y < nc; y++) {
				long index = dParameters_.getAttributeIndex(u1, u1val, y);
				xcount += dParameters_.getCountAtFullIndex(index);
			}

			for (int c = 0; c < nc; c++) {
				int avyCount = (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, c));

				int ycount = (int) dParameters_.getCountAtFullIndex(c);

				if (avyCount > 0) {
					m += (avyCount / N) * Math.log( avyCount / ( xcount/N * ycount ) ) / Math.log(2);
				}
			}
		}

		return m;
	}

	public static double computeMutualInformationPerFeatureValue(int u1, int u1valin, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		int xcount = 0;
		int xNOTcount = 0;
		int[] avyCount = new int[nc];
		int[] avyNOTcount = new int[nc];

		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			if (u1val == u1valin) {
				for (int y = 0; y < nc; y++) {
					long index = dParameters_.getAttributeIndex(u1, u1val, y);
					xcount += dParameters_.getCountAtFullIndex(index);

					avyCount[y] += (int) dParameters_.getCountAtFullIndex(index);
				}

				//for (int c = 0; c < nc; c++) {
				//	avyCount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, c));
				//}
			} else {
				for (int y = 0; y < nc; y++) {
					long index = dParameters_.getAttributeIndex(u1, u1val, y);
					xNOTcount += dParameters_.getCountAtFullIndex(index);

					avyNOTcount[y] += (int) dParameters_.getCountAtFullIndex(index);
				}

				//for (int c = 0; c < nc; c++) {
				//	avyNOTcount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, c));
				//}
			}
		}

		double m = 0;
		for (int c = 0; c < nc; c++) {
			int ycount = (int) dParameters_.getCountAtFullIndex(c);

			if (avyCount[c] > 0) {
				m += (avyCount[c] / N) * Math.log( avyCount[c] / ( xcount/N * ycount ) ) / Math.log(2);
			}

			if (avyNOTcount[c] > 0) {
				m += (avyNOTcount[c] / N) * Math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / Math.log(2);
			}
		}

		return m;
	}

	public static double computeMutualInformation(int u1, int u2, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		double m = 0;
		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {

				int xcount = 0;
				for (int y = 0; y < nc; y++) {
					long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, y);
					xcount += dParameters_.getCountAtFullIndex(index);
				}

				for (int c = 0; c < nc; c++) {
					int avyCount = (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, c));

					int ycount = (int) dParameters_.getCountAtFullIndex(c);

					if (avyCount > 0) {
						m += (avyCount / N) * Math.log( avyCount / ( xcount/N * ycount ) ) / Math.log(2);
					}
				}
			}
		}

		//		if (m > 1.0) {
		//			System.out.println("Something wrong, MI can't be greater than 1.0");
		//			System.exit(-1);
		//		}

		return m;
	}

	public static double computeMutualInformationPerFeatureValue(int u1, int u1valin, int u2, int u2valin, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		int xcount = 0;
		int xNOTcount = 0;
		int[] avyCount = new int[nc];
		int[] avyNOTcount = new int[nc];

		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				if (u1val == u1valin && u2val == u2valin) {
					for (int y = 0; y < nc; y++) {
						long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, y);
						xcount += dParameters_.getCountAtFullIndex(index);

						avyCount[y] += (int) dParameters_.getCountAtFullIndex(index);
					}

					//for (int c = 0; c < nc; c++) {
					//	avyCount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, c));
					//}
				} else {
					for (int y = 0; y < nc; y++) {
						long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, y);
						xNOTcount += dParameters_.getCountAtFullIndex(index);

						avyNOTcount[y] += (int) dParameters_.getCountAtFullIndex(index);
					}

					//for (int c = 0; c < nc; c++) {
					//	avyNOTcount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, c));
					//}
				}
			}
		}

		double m = 0;
		for (int c = 0; c < nc; c++) {
			int ycount = (int) dParameters_.getCountAtFullIndex(c);

			if (avyCount[c] > 0) {
				m += (avyCount[c] / N) * Math.log( avyCount[c] / ( xcount/N * ycount ) ) / Math.log(2);
			}

			if (avyNOTcount[c] > 0) {
				m += (avyNOTcount[c] / N) * Math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / Math.log(2);
			}
		}

		return m;
	}

	public static double computeMutualInformation(int u1, int u2, int u3, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		double m = 0;
		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				for (int u3val = 0; u3val < dParameters_.getParamsPerAtt()[u3]; u3val++) {

					int xcount = 0;
					for (int y = 0; y < nc; y++) {
						long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y);
						xcount += dParameters_.getCountAtFullIndex(index);
					}

					for (int c = 0; c < nc; c++) {
						int avyCount = (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c));

						int ycount = (int) dParameters_.getCountAtFullIndex(c);

						if (avyCount > 0) {
							m += (avyCount / N) * Math.log( avyCount / ( xcount/N * ycount ) ) / Math.log(2);
						}
					}
				}
			}
		}

		//		if (m > 1.0) {
		//			System.out.println("Econtered m = " + m);
		//			System.out.println("Something wrong, MI can't be greater than 1.0");
		//			System.exit(-1);
		//		}

		return m;
	}

	public static double computeMutualInformationPerFeatureValue(int u1, int u1valin, int u2, int u2valin, int u3, int u3valin, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		int xcount = 0;
		int xNOTcount = 0;
		int[] avyCount = new int[nc];
		int[] avyNOTcount = new int[nc];

		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				for (int u3val = 0; u3val < dParameters_.getParamsPerAtt()[u3]; u3val++) {
					if (u1val == u1valin && u2val == u2valin && u3val == u3valin) {
						for (int y = 0; y < nc; y++) {
							long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y);
							xcount += dParameters_.getCountAtFullIndex(index);

							avyCount[y] += (int) dParameters_.getCountAtFullIndex(index);
						}

						//for (int c = 0; c < nc; c++) {
						//	avyCount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c));
						//}
					} else {
						for (int y = 0; y < nc; y++) {
							long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y);
							xNOTcount += dParameters_.getCountAtFullIndex(index);

							avyNOTcount[y] += (int) dParameters_.getCountAtFullIndex(index);
						}

						//for (int c = 0; c < nc; c++) {
						//	avyNOTcount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val,  c));
						//}
					}
				}
			}
		}

		double m = 0;
		for (int c = 0; c < nc; c++) {
			int ycount = (int) dParameters_.getCountAtFullIndex(c);

			if (avyCount[c] > 0) {
				m += (avyCount[c] / N) * Math.log( avyCount[c] / ( xcount/N * ycount ) ) / Math.log(2);
			}

			if (avyNOTcount[c] > 0) {
				m += (avyNOTcount[c] / N) * Math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / Math.log(2);
			}
		}

		return m;
	}

	public static double computeMutualInformation(int u1, int u2, int u3, int u4, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		double m = 0;
		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				for (int u3val = 0; u3val < dParameters_.getParamsPerAtt()[u3]; u3val++) {
					for (int u4val = 0; u4val < dParameters_.getParamsPerAtt()[u4]; u4val++) {

						int xcount = 0;
						for (int y = 0; y < nc; y++) {
							long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y);
							xcount += dParameters_.getCountAtFullIndex(index);
						}

						for (int c = 0; c < nc; c++) {
							int avyCount = (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c));

							int ycount = (int) dParameters_.getCountAtFullIndex(c);

							if (avyCount > 0) {
								m += (avyCount / N) * Math.log( avyCount / ( xcount/N * ycount ) ) / Math.log(2);
							}
						}
					}
				}
			}
		}

		return m;
	}

	public static double computeMutualInformationPerFeatureValue(int u1, int u1valin, int u2, int u2valin, int u3, int u3valin, int u4, int u4valin, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		int xcount = 0;
		int xNOTcount = 0;
		int[] avyCount = new int[nc];
		int[] avyNOTcount = new int[nc];

		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				for (int u3val = 0; u3val < dParameters_.getParamsPerAtt()[u3]; u3val++) {
					for (int u4val = 0; u4val < dParameters_.getParamsPerAtt()[u4]; u4val++) {
						if (u1val == u1valin && u2val == u2valin && u3val == u3valin && u4val == u4valin) {
							for (int y = 0; y < nc; y++) {
								long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y);
								xcount += dParameters_.getCountAtFullIndex(index);

								avyCount[y] += (int) dParameters_.getCountAtFullIndex(index);
							}

							//for (int c = 0; c < nc; c++) {
							//	avyCount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c));
							//}
						} else {
							for (int y = 0; y < nc; y++) {
								long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y);
								xNOTcount += dParameters_.getCountAtFullIndex(index);

								avyNOTcount[y] += (int) dParameters_.getCountAtFullIndex(index);
							}

							//for (int c = 0; c < nc; c++) {
							//	avyNOTcount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c));
							//}
						}
					}
				}
			}
		}

		double m = 0;
		for (int c = 0; c < nc; c++) {
			int ycount = (int) dParameters_.getCountAtFullIndex(c);

			if (avyCount[c] > 0) {
				m += (avyCount[c] / N) * Math.log( avyCount[c] / ( xcount/N * ycount ) ) / Math.log(2);
			}

			if (avyNOTcount[c] > 0) {
				m += (avyNOTcount[c] / N) * Math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / Math.log(2);
			}
		}

		return m;
	}

	public static double computeMutualInformation(int u1, int u2, int u3, int u4, int u5, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		double m = 0;
		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				for (int u3val = 0; u3val < dParameters_.getParamsPerAtt()[u3]; u3val++) {
					for (int u4val = 0; u4val < dParameters_.getParamsPerAtt()[u4]; u4val++) {
						for (int u5val = 0; u5val < dParameters_.getParamsPerAtt()[u5]; u5val++) {

							int xcount = 0;
							for (int y = 0; y < nc; y++) {
								long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y);
								xcount += dParameters_.getCountAtFullIndex(index);
							}

							for (int c = 0; c < nc; c++) {
								int avyCount = (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c));

								int ycount = (int) dParameters_.getCountAtFullIndex(c);

								if (avyCount > 0) {
									m += (avyCount / N) * Math.log( avyCount / ( xcount/N * ycount ) ) / Math.log(2);
								}
							}
						}
					}
				}
			}
		}

		return m;
	}

	public static double computeMutualInformationPerFeatureValue(int u1, int u1valin, int u2, int u2valin, int u3, int u3valin, int u4, int u4valin, int u5, int u5valin, Parameter dParameters_) {

		int nc = dParameters_.getNC();
		double N = dParameters_.getN();

		int xcount = 0;
		int xNOTcount = 0;
		int[] avyCount = new int[nc];
		int[] avyNOTcount = new int[nc];

		for (int u1val = 0; u1val < dParameters_.getParamsPerAtt()[u1]; u1val++) {
			for (int u2val = 0; u2val < dParameters_.getParamsPerAtt()[u2]; u2val++) {
				for (int u3val = 0; u3val < dParameters_.getParamsPerAtt()[u3]; u3val++) {
					for (int u4val = 0; u4val < dParameters_.getParamsPerAtt()[u4]; u4val++) {
						for (int u5val = 0; u5val < dParameters_.getParamsPerAtt()[u5]; u5val++) {
							if (u1val == u1valin && u2val == u2valin && u3val == u3valin && u4val == u4valin && u5val == u5valin) {
								for (int y = 0; y < nc; y++) {
									long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y);
									xcount += dParameters_.getCountAtFullIndex(index);

									avyCount[y] += (int) dParameters_.getCountAtFullIndex(index);
								}

								//for (int c = 0; c < nc; c++) {
								//	avyCount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c));
								//}
							} else {
								for (int y = 0; y < nc; y++) {
									long index = dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y);
									xNOTcount += dParameters_.getCountAtFullIndex(index);

									avyNOTcount[y] += (int) dParameters_.getCountAtFullIndex(index);
								}

								//for (int c = 0; c < nc; c++) {
								//	avyNOTcount[c] += (int) dParameters_.getCountAtFullIndex(dParameters_.getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c));
								//}
							}
						}
					}
				}
			}
		}

		double m = 0;
		for (int c = 0; c < nc; c++) {
			int ycount = (int) dParameters_.getCountAtFullIndex(c);

			if (avyCount[c] > 0) {
				m += (avyCount[c] / N) * Math.log( avyCount[c] / ( xcount/N * ycount ) ) / Math.log(2);
			}

			if (avyNOTcount[c] > 0) {
				m += (avyNOTcount[c] / N) * Math.log( avyNOTcount[c] / ( xNOTcount/N * ycount ) ) / Math.log(2);
			}
		}

		return m;
	}

	public static Map<String, Double> sort(Map<String, Double> unsortMap) {

		boolean ASC = true;
		boolean DESC = false;

		//printMap(unsortMap);

		Map<String, Double> sortedMapAsc = sortByComparator(unsortMap, DESC);

		//printMap(sortedMapAsc);

		return sortedMapAsc;
	}

	private static Map<String, Double> sortByComparator(Map<String, Double> unsortMap, final boolean order) {

		List<Entry<String, Double>> list = new LinkedList<Entry<String, Double>>(unsortMap.entrySet());

		// Sorting the list based on values
		Collections.sort(list, new Comparator<Entry<String, Double>>()
		{
			public int compare(Entry<String, Double> o1,
					Entry<String, Double> o2)
			{
				if (order)
				{
					return o1.getValue().compareTo(o2.getValue());
				}
				else
				{
					return o2.getValue().compareTo(o1.getValue());

				}
			}
		});

		// Maintaining insertion order with the help of LinkedList
		Map<String, Double> sortedMap = new LinkedHashMap<String, Double>();
		for (Entry<String, Double> entry : list)
		{
			sortedMap.put(entry.getKey(), entry.getValue());
		}

		return sortedMap;
	}

	public static void printMap(Map<String, Double> map) {

		System.out.println("----------------------------------------------------");
		for (Entry<String, Double> entry : map.entrySet()) {
			System.out.println("Key : " + entry.getKey() + " Value : "+ entry.getValue());
		}
		System.out.println("----------------------------------------------------");

	}

	public static int numberOfCharInString(String key, char c) {
		int num = 0;
		for (int i = 0; i < key.length(); i++) {
			if (key.charAt(i) == c) {
				num++;
			}
		}
		return num;
	}

	public static String[] getStringFromLine(String val, char delimiter) {

		//val = val.replaceAll("[{()}]", "");


		//String[] parseFlowValues = val.split("\\s*,\\s*");
		String[] parseValues = val.split("\\s*" + delimiter + "\\s*",-1);

		int numValues = parseValues.length;

		String[] valuesString = null;

		if (numValues != 0) {
			valuesString = new String[numValues];

			for (int i = 0; i < parseValues.length; i++) {
				valuesString[i] = parseValues[i];
			}
		}

		return valuesString;
	}

	public static double[] getDoubleFromLine(String val, char delimiter) {

		val = val.replaceAll("[{()}]", "");

		//String[] parseFlowValues = val.split("\\s*,\\s*");
		String[] parseValues = val.split("\\s*" + delimiter + "\\s*");

		int numValues = parseValues.length;

		double[] valuesDouble = null;

		if (numValues != 0) {
			valuesDouble = new double[numValues];

			for (int i = 0; i < parseValues.length; i++) {
				valuesDouble[i] = Double.parseDouble(parseValues[i]);
			}
		}

		return valuesDouble;
	}

	public static int[] getIntegerFromLine(String val, char delimiter) {

		int[] valuesInt = null;

		if (val.equalsIgnoreCase("{}") || val.equalsIgnoreCase("")) {
			return valuesInt;
		}

		val = val.replaceAll("[{()}]", "");

		//String[] parseFlowValues = val.split("\\s*,\\s*");
		String[] parseValues = val.split("\\s*" + delimiter + "\\s*");

		int numValues = parseValues.length;

		if (numValues != 0) {
			valuesInt = new int[numValues];

			for (int i = 0; i < parseValues.length; i++) {
				valuesInt[i] = Integer.parseInt(parseValues[i]);
			}
		}

		return valuesInt;
	}

	public static boolean[] getBooleanFromLine(String val) {

		val = val.replaceAll("[{()}]", "");

		String[] parseValues = val.split("\\s*,\\s*");

		int numValues = parseValues.length;

		boolean[] isNumericFlag = null;

		if (numValues != 0) {
			isNumericFlag = new boolean[numValues];

			for (int i = 0; i < parseValues.length; i++) {
				int flag = Integer.parseInt(parseValues[i]);
				if (flag == 0)
					isNumericFlag[i] = false;
				else 
					isNumericFlag[i] = true;
			}
		}

		return isNumericFlag;
	}

	public static int[] getIndices(String key) {
		int[] vals;
		String delimiter = ":|\\,";
		String[] parseValues = key.split("\\s*" + delimiter + "\\s*");

		vals = new int[parseValues.length];
		int j = 0;
		for (int i = 0; i < parseValues.length; i++) {
			vals[j] = Integer.parseInt(parseValues[i]);
			j++;
		}

		return vals;
	}

	public static Map<Set<Integer>, Double> sortSet(Map<Set<Integer>, Double> unsortMap) {

		boolean ASC = true;
		boolean DESC = false;

		//printMap(unsortMap);

		Map<Set<Integer>, Double> sortedMapAsc = sortSetByComparator(unsortMap, DESC);

		//printMap(sortedMapAsc);

		return sortedMapAsc;
	}

	private static Map<Set<Integer>, Double> sortSetByComparator(Map<Set<Integer>, Double> unsortMap, final boolean order) {

		List<Entry<Set<Integer>, Double>> list = new LinkedList<Entry<Set<Integer>, Double>>(unsortMap.entrySet());

		// Sorting the list based on values
		Collections.sort(list, new Comparator<Entry<Set<Integer>, Double>>() {
			public int compare(Entry<Set<Integer>, Double> o1, Entry<Set<Integer>, Double> o2) {
				if (order) {
					return o1.getValue().compareTo(o2.getValue());
				} else {
					return o2.getValue().compareTo(o1.getValue());

				}
			}
		});

		// Maintaining insertion order with the help of LinkedList
		Map<Set<Integer>, Double> sortedMap = new LinkedHashMap<Set<Integer>, Double>();
		for (Entry<Set<Integer>, Double> entry : list) {
			sortedMap.put(entry.getKey(), entry.getValue());
		}

		return sortedMap;
	}

	public static File addNoise(int numNoiseColumns, File sourceFile) throws FileNotFoundException, IOException {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), Globals.getBUFFER_SIZE()), Globals.getARFF_BUFFER_SIZE());
		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		File out = File.createTempFile("trainCV-", ".arff", new File(Globals.getTempDirectory()));
		System.out.println("(SUtils, addNoise()) Creating File at: " +  out.getAbsolutePath());
		out.deleteOnExit();

		Writer w = new BufferedWriter(new FileWriter(out));

		String header = "";
		header += "@relation '" + "contrieved" + "'\n\n";
		for (int i = 0; i < structure.numAttributes() - 1; i++) {
			header += "@attribute x" + i + " { ";
			for (int j = 0; j < structure.attribute(i).numValues(); j++) {
				if (j == structure.attribute(i).numValues() - 1) {
					header += structure.attribute(i).value(j);
					//header += (double) j;
				} else {
					header += structure.attribute(i).value(j) + ", ";
					//header += (double) j + ", ";
				}
			}
			header += " }\n";
		}

		for (int i = 0; i < numNoiseColumns; i++) {
			header += "@attribute x" + (i +  (structure.numAttributes() - 1)) + " {0, 1, 2}\n"; 
			//header += "@attribute x" + (i +  (structure.numAttributes() - 1)) + " {0, 1}\n";
		}

		int classIndex = structure.classIndex();
		header += "@attribute x" + (classIndex + numNoiseColumns) + " { ";
		for (int j = 0; j < structure.attribute(classIndex).numValues(); j++) {
			if (j == structure.attribute(classIndex).numValues() - 1) {
				header += structure.attribute(classIndex).value(j);
			} else {
				header += structure.attribute(classIndex).value(j) + ", ";
			}
		}
		header += " }\n";


		header += "\n@data\n\n";

		w.write(header);

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(System.currentTimeMillis());
		RandomDataGenerator r = new RandomDataGenerator(rg);

		Instance current = null;
		while ((current = reader.readInstance(structure)) != null) {

			for (int u = 0; u < structure.numAttributes() - 1; u++) {
				w.write(current.attribute(u).value((int)current.value(u))+ ",");
			}

			for (int i = 0; i < numNoiseColumns; i++) {
				w.write(r.nextSecureInt(0, 2)+",");
				//w.write(r.nextSecureInt(0, 1)+",");
			}

			w.write(current.attribute(structure.numAttributes()-1).value((int) current.value(structure.numAttributes()-1)));
			w.write("\n");
		}

		w.close();

		return out;
	}

	public static int sampleFromNonUniformDistribution(double[] ds, RandomDataGenerator r) {

		double rand = r.nextUniform(0.0, 1.0);
		int chosenVal = 0;
		double sumProbs = ds[chosenVal];
		
		while (rand > sumProbs) {
			chosenVal++;
			sumProbs += ds[chosenVal];
		}
		
		return chosenVal;
	}

	public static double getArrayListDoubleMean(ArrayList<Double> list) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public static double getArrayListDoubleVariance(ArrayList<Double> list) {
		// TODO Auto-generated method stub
		return 0;
	}

	public static double getArrayListDoubleMin(ArrayList<Double> list) {
		// TODO Auto-generated method stub
		return 0;
	}

	public static double getArrayListDoubleMax(ArrayList<Double> list) {
		// TODO Auto-generated method stub
		return 0;
	}

	public static double getArrayListDoubleMode(ArrayList<Double> list) {
		// TODO Auto-generated method stub
		return 0;
	}

}


