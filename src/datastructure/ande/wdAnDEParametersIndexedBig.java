package datastructure.ande;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.BitSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import utils.Globals;
import weka.core.Instance;
import weka.core.Instances;

public class wdAnDEParametersIndexedBig extends wdAnDEParameters {

	protected final static int SENTINEL = -1;
	protected final static double PROBA_VALUE_WHEN_ZERO_COUNT = -25; //in Log Scale
	protected final static double GRADIENT_VALUE_WHEN_ZERO_COUNT = 0.0;

	int[][] indexes;
	int actualNumberParameters;

	BitSet[] combinationRequired;

	private int nLines;	
	private int remainders;

	public wdAnDEParametersIndexedBig() throws FileNotFoundException, IOException {

		super();

		if (Globals.isVerbose()) {
			System.out.print("In Constructor of wdAnDEParametersIndexedBig(), np = " + np);
		}

		if (np <= MAX_TAB_LENGTH) {
			System.out.println(" -- WARNING: The number of parameters is not that big, it would be faster to use 'wdAnDEParametersFlat'.");
		}

		nLines = (int) (np / MAX_TAB_LENGTH) + 1;
		remainders = (int) (np % MAX_TAB_LENGTH);

		if (Globals.isVerbose()) {
			System.out.println("[Will be needing " + (int) (np / MAX_TAB_LENGTH) + " rows of size: " + MAX_TAB_LENGTH + " and last row will be of size: " + remainders +"]");
		}

		combinationRequired = new BitSet[nLines];
		for (int l = 0; l < combinationRequired.length - 1; l++) {
			combinationRequired[l] = new BitSet(MAX_TAB_LENGTH);
		}		
		combinationRequired[combinationRequired.length - 1] = new BitSet(remainders);
	}

	@Override
	public void updateFirstPass(Instance inst) {

		int x_C = (int) inst.classValue();
		setCombinationRequired(x_C);

		N++;

		if (level == 0) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				setCombinationRequired(index);
			}

		} else if (level == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				setCombinationRequired(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					setCombinationRequired(index);					
				}
			}

		} else if (level == 2) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				setCombinationRequired(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					setCombinationRequired(index);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						setCombinationRequired(index);						
					}
				}
			}
		} else if (level == 3) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				setCombinationRequired(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					setCombinationRequired(index);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						setCombinationRequired(index);

						for (int u4 = 0; u4 < u3; u4++) {
							int x_u4 = (int) inst.value(u4);

							index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
							setCombinationRequired(index);
						}
					}
				}
			}
		} else if (level == 4) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				setCombinationRequired(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					setCombinationRequired(index);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						setCombinationRequired(index);

						for (int u4 = 0; u4 < u3; u4++) {
							int x_u4 = (int) inst.value(u4);

							index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
							setCombinationRequired(index);

							for (int u5 = 0; u5 < u4; u5++) {
								int x_u5 = (int) inst.value(u5);

								index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C);
								setCombinationRequired(index);
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void finishedFirstPass() {

		indexes = new int[nLines][];
		for (int l = 0; l < indexes.length - 1; l++) {
			indexes[l] = new int[MAX_TAB_LENGTH];
		}
		indexes[indexes.length - 1] = new int[remainders];

		actualNumberParameters = 0;

		for (int l = 0; l < indexes.length; l++) {
			for (int i = 0; i < indexes[l].length; i++) {
				if (combinationRequired[l].get(i)) {
					indexes[l][i] = actualNumberParameters;
					actualNumberParameters++;
				} else {
					indexes[l][i] = SENTINEL;
				}				
			}
			combinationRequired[l] = null;
		}

		System.out.println("	Original number of parameters: " + np + " (" + np/(1024*1024*1024) + "gb)");
		System.out.println("	Compressed number of parameters: " + actualNumberParameters + " (" + actualNumberParameters/(1024*1024*1024) + "gb)");
		double ratio = actualNumberParameters/(double)np;
		System.out.println("	Compression of: " + ratio  + "");
		
		// now compress count table
		allocateMemoryForCountsParametersProbabilitiesAndGradients(actualNumberParameters);	
	}

	@Override
	public boolean needSecondPass() {
		return true;
	}

	@Override
	public void updateAfterFirstPass(Instance inst) {
		int x_C = (int) inst.classValue();
		incCountAtFullIndex(x_C);

		if (level == 0) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);
			}

		} else if (level == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);
				}
			}

		} else if (level == 2) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						incCountAtFullIndex(index);
					}
				}
			}
		} else if (level == 3) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						incCountAtFullIndex(index);

						for (int u4 = 0; u4 < u3; u4++) {
							int x_u4 = (int) inst.value(u4);

							index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
							incCountAtFullIndex(index);
						}
					}
				}
			}
		} else if (level == 4) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						incCountAtFullIndex(index);

						for (int u4 = 0; u4 < u3; u4++) {
							int x_u4 = (int) inst.value(u4);

							index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
							incCountAtFullIndex(index);

							for (int u5 = 0; u5 < u4; u5++) {
								int x_u5 = (int) inst.value(u5);

								index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C);
								incCountAtFullIndex(index);
							}
						}
					}
				}
			}
		}
	}


	@Override
	public double getCountAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact == SENTINEL) {
			return 0;
		} else {
			return xyCount[indexCompact];
		}
	}

	public void incCountAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact]++;
		}
	}

	public int getIndexCompact(long index) {
		int indexL = (int) (index / MAX_TAB_LENGTH);
		int indexC = (int) (index % MAX_TAB_LENGTH);
		return indexes[indexL][indexC];
	}

	public void setCombinationRequired(long index) {
		int indexL = (int) (index / MAX_TAB_LENGTH);
		int indexC = (int) (index % MAX_TAB_LENGTH);
		combinationRequired[indexL].set(indexC);
	}


} // ends class

