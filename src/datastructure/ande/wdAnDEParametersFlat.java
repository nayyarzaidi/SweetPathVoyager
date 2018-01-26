package datastructure.ande;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.Queue;

import utils.Globals;
import weka.core.Instance;

public class wdAnDEParametersFlat extends wdAnDEParameters {

	private Queue<Instance> queue = new LinkedList<Instance>();
	private int counter = 0;

	public wdAnDEParametersFlat() throws FileNotFoundException, IOException {

		super();

		if (Globals.isVerbose()) {
			System.out.print("In the Constructor of wdAnDEParametersFlat(), np = " + np);
		}

		if (np > MAX_TAB_LENGTH) {
			System.err.println("CRITICAL ERROR: --structureParameter: 'Flat' not implemented for such dimensionalities. Use 'IndexedBig' or 'BitMap' or 'Hash'");
			System.exit(-1);
		}

		allocateMemoryForCountsParametersProbabilitiesAndGradients(np);	
	}

	@Override
	public void updateFirstPass(Instance inst) {
		
		//System.out.println(Arrays.toString(xyCount));

		if (adaptiveControl.equalsIgnoreCase("Decay")) {

			applyDecay();

		} else if (adaptiveControl.equalsIgnoreCase("Window")) {

			if (counter < adaptiveControlParameter) {
				queue.add(inst);
			} else {
				unUpdateFirstPass(queue.remove());
				queue.add(inst);
			}

			counter++;
		}

		int x_C = (int) inst.classValue();
		xyCount[x_C]++;

		N++;

		if (level == 0) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);				
			}

		} else if (level == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);	

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);		
				}
			}

		} else if (level == 2) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);		

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);	

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						incCountAtFullIndex(index);		
					}
				}
			}
		}

	}

	private void unUpdateFirstPass(Instance inst) {

		int x_C = (int) inst.classValue();
		xyCount[x_C]--;

		N--;

		if (level == 0) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				decCountAtFullIndex(index);				
			}

		} else if (level == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				decCountAtFullIndex(index);	

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					decCountAtFullIndex(index);		
				}
			}

		} else if (level == 2) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				decCountAtFullIndex(index);		

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					decCountAtFullIndex(index);	

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						decCountAtFullIndex(index);		
					}
				}
			}
		}

	}

	private void applyDecay() {
		double a = adaptiveControlParameter;

		for (int i = 0; i < np; i++) {
			xyCount[i] = (xyCount[i] * Math.exp(-a));
		}

		N = N * Math.exp(-a);
	}

	@Override
	public void finishedFirstPass() {		
		// Nothing to do here.
	}

	@Override
	public boolean needSecondPass() {
		return false;
	}

	@Override
	public void updateAfterFirstPass(Instance inst) {
		// Nothing to do, needSecondPass() is false.
	}	

	@Override
	public double getCountAtFullIndex(long index) {
		return xyCount[(int)index];
	}

	//public void initCount(long size) {
	//	xyCount = new int[(int)size];
	//}

	public void incCountAtFullIndex(long index) {

		//		if (adaptiveControl.equalsIgnoreCase("prequential") || 
		//				experimentType.equalsIgnoreCase("flowMachines") || 
		//				experimentType.equalsIgnoreCase("drift")) {
		//
		//			if (adaptiveControl.equalsIgnoreCase("Decay")) {
		//
		//				double a = adaptiveControlParameter;
		//				double ns = (Duration.between(Instant.now(), timestamp[(int)index]).getNano())/Math.pow(10, 9);
		//						
		//				//xyCount[(int)index] = xyCount[(int)index] * Math.pow(a, ns);
		//				
		//				xyCount[(int)index] = xyCount[(int)index] * Math.exp(-a * ns);
		//				timestamp[(int)index] = Instant.now();
		//			}
		//
		//		} else {

		xyCount[(int)index]++;

		//		}
	}

	public void decCountAtFullIndex(long index) {
		xyCount[(int)index]--;
	}


} // ends class

