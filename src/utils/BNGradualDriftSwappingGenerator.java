package utils;

/*
 * Created by Lee on 6/09/2015.
 */

import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

import java.util.Arrays;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

public class BNGradualDriftSwappingGenerator extends DriftGenerator {
	private static final long serialVersionUID = 1291115908166720203L;

	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 1000, 0,
					Integer.MAX_VALUE);

	protected InstancesHeader streamHeader;

	/**
	 * p(x) before drift
	 */
	double[][] px;
	/**
	 * p(y|x) before drift (only kept to be sure we're actually swapping something)
	 */
	double[][] pygxInit;

	/**
	 * p(y|x) during and after drift
	 */
	double[][] pygxDrifting;

	RandomDataGenerator r;

	long nInstancesGeneratedSoFar;

	private int nLinesToChange;

	private int[] linesToChange;
	private int nLinesChanged;

	@Override
	public long estimatedRemainingInstances() {
		return -1;
	}

	@Override
	public boolean hasMoreInstances() {
		return true;
	}

	@Override
	public boolean isRestartable() {
		return false;
	}

	@Override
	public void restart() {

	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {

	}

	@Override
	public String getPurposeString() {
		return "Generates a stream with an abrupt drift of given magnitude.";
	}

	@Override
	public InstancesHeader getHeader() {
		return streamHeader;
	}

	protected void generateHeader() {

		FastVector<Attribute> attributes = getHeaderAttributes(nAttributes.getValue(), nValuesPerAttribute.getValue());

		this.streamHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
	}

	@Override
	public InstanceExample nextInstance() {


		if (nInstancesGeneratedSoFar > burnInNInstances.getValue() && nInstancesGeneratedSoFar <= burnInNInstances.getValue() + driftLength.getValue()) {
			//we only care to swap lines if during drift
			
			long nInstancesInDrift = nInstancesGeneratedSoFar-burnInNInstances.getValue();
			long driftLength = this.driftLength.getValue();
			
			long nLinesShouldHaveSwapped = Math.round(1.0*nLinesToChange*nInstancesInDrift/driftLength);
			
			long nLinesToSwapThisInstance = nLinesShouldHaveSwapped-nLinesChanged;
			int copyNLinesChanged = nLinesChanged;

			for (int i = copyNLinesChanged; i < (nLinesToSwapThisInstance + copyNLinesChanged); i++) {
				int lineNo = linesToChange[i];
				pygxDrifting[lineNo] = new double[nValuesPerAttribute.getValue()];
				double[] lineCPT = pygxDrifting[lineNo];
				int chosenClass;

				do {
					chosenClass = r.nextInt(0, lineCPT.length - 1);
					// making sure we choose a different class value
				} while (pygxInit[lineNo][chosenClass] == 1.0);

				for (int c = 0; c < lineCPT.length; c++) {
					if (c == chosenClass) {
						lineCPT[c] = 1.0;
					} else {
						lineCPT[c] = 0.0;
					}
				}
				nLinesChanged++;
			}
		}
		
		if (nInstancesGeneratedSoFar > burnInNInstances.getValue() + driftLength.getValue()){
			if(nLinesChanged<nLinesToChange){
				throw new RuntimeException("Should have swapped "+nLinesToChange+" - actually done "+nLinesChanged);
			}
		}
		

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		int[] indexes = new int[nAttributes.getValue()];

		// setting values for x_1,...,x_n
		for (int a = 0; a < indexes.length; a++) {
			// choosing values of x_1,...,x_n
			double rand = r.nextUniform(0.0, 1.0, true);
			int chosenVal = 0;
			double sumProba = px[a][chosenVal];
			while (rand > sumProba) {
				chosenVal++;
				sumProba += px[a][chosenVal];
			}
			indexes[a] = chosenVal;
			inst.setValue(a, chosenVal);
		}

		int lineNoCPT = getIndex(indexes);

		int chosenClassValue = 0;
		while (pygxDrifting[lineNoCPT][chosenClassValue] != 1.0) {
			chosenClassValue++;
		}
		inst.setClassValue(chosenClassValue);

		nInstancesGeneratedSoFar++;
		// System.out.println("generated "+inst);
		return new InstanceExample(inst);
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		System.out.println("burnIn=" + burnInNInstances.getValue());
		generateHeader();

		int nCombinationsValuesForPX = 1;
		for (int a = 0; a < nAttributes.getValue(); a++) {
			nCombinationsValuesForPX *= nValuesPerAttribute.getValue();
		}

		px = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
		pygxInit = new double[nCombinationsValuesForPX][nValuesPerAttribute.getValue()];

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);

		// generating distribution before drift

		// p(x)
		generateRandomPx(px, r);

		// p(y|x)
		generateRandomPyGivenX(pygxInit, r);

		if (driftPriors.isSet()) {
			throw new RuntimeException("Drifiting priors not implemented for this generator");
		}

		// conditional
		if (driftConditional.isSet()) {
			pygxDrifting = new double[nCombinationsValuesForPX][];
			for (int line = 0; line < pygxDrifting.length; line++) {
				// default is same distrib
				pygxDrifting[line] = pygxInit[line];
			}

			nLinesToChange = (int) Math.round(driftMagnitudeConditional.getValue() * nCombinationsValuesForPX);
			if (nLinesToChange == 0.0) {
				System.out.println("Not enough drift to be noticeable in p(y|x) - unchanged");
				pygxDrifting = pygxInit;

			} else {
				linesToChange = r.nextPermutation(nCombinationsValuesForPX, nLinesToChange);

				nLinesChanged = 0;

				// copy pygxbd into pygxDrifting
				for (int line = 0; line < pygxInit.length; line++) {
					pygxDrifting[line] = Arrays.copyOf(pygxInit[line], pygxInit[line].length);
				}

				System.out.println("exact magnitude for p(y|x)=" + computeMagnitudePYGX(pygxInit, pygxDrifting) + "\tasked="
								+ driftMagnitudeConditional.getValue());

			}
		} else {
			pygxDrifting = pygxInit;
		}

		// System.out.println(Arrays.toString(pxbd));
		// System.out.println(Arrays.toString(pxad));

		nInstancesGeneratedSoFar = 0L;

	}

	protected final int getIndex(int... indexes) {
		int index = indexes[0];
		for (int i = 1; i < indexes.length; i++) {
			index *= nValuesPerAttribute.getValue();
			index += indexes[i];
		}
		return index;

	}

}
