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

public class BNGradualDriftMixtureGenerator extends DriftGenerator {

	private static final long serialVersionUID = 1291115908166720203L;

	//	//public IntOption nAttributes = new IntOption("nAttributes", 'n',
	//	//		"Number of attributes as parents of the class", 2, 1, 10);
	//	//public IntOption nValuesPerAttribute = new IntOption("nValuesPerAttribute",
	//	//		'v', "Number of values per attribute", 2, 2, 5);
	//	public IntOption burnInNInstances = new IntOption("burnInNInstances", 'b',
	//			"Number of instances before the start of the drift", 10000, 1,
	//			Integer.MAX_VALUE);
	//	public FloatOption driftMagnitude = new FloatOption(
	//			"driftMagnitude",
	//			'm',
	//			"Magnitude of the drift between the starting probability and the one after the drift."
	//					+ " Magnitude is expressed as the Hellinger distance [0,1]",
	//					0.5, 1e-20, 0.9);
	//	public FloatOption precisionDriftMagnitude = new FloatOption(
	//			"epsilon",
	//			'e',
	//			"Precision of the drift magnitude for p(x) (how far from the set magnitude is acceptable)",
	//			0.01, 1e-20, 1.0);

	//	public FlagOption driftConditional = new FlagOption("driftConditional",
	//			'c',
	//			"States if the drift should apply to the conditional distribution p(y|x).");
	//	public FlagOption driftPriors = new FlagOption("driftPriors", 'p',
	//			"States if the drift should apply to the prior distribution p(x). ");
	//	public IntOption seed = new IntOption("seed", 'r',
	//			"Seed for random number generator", -1, Integer.MIN_VALUE,
	//			Integer.MAX_VALUE);

	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000);

	protected InstancesHeader streamHeader;

	/**
	 * p(x) before drift
	 */
	double[][] pxbd;

	/**
	 * p(y|x) before drift
	 */
	double[][] pygxbd;

	/**
	 * p(x) after drift
	 */
	double[][] pxad;

	/**
	 * p(y|x) after drift
	 */
	double[][] pygxad;

	RandomDataGenerator r;

	long nInstancesGeneratedSoFar;

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
		double[][] px;
		double[][] pygx;

		double probSecondDistrib;

		if (nInstancesGeneratedSoFar <= burnInNInstances.getValue()) {
			probSecondDistrib = 0.0;
		} else if (nInstancesGeneratedSoFar > burnInNInstances.getValue() + driftLength.getValue()) {
			probSecondDistrib = 1.0;
		} else {
			// prob proportional to where we're at in the drift
			int nInstancesInDrift = (int) (nInstancesGeneratedSoFar - burnInNInstances.getValue());
			probSecondDistrib = 1.0 * nInstancesInDrift / driftLength.getValue();
		}

		boolean isSecondDistrib = r.nextUniform(0.0, 1.0, true) <= probSecondDistrib;

		if (isSecondDistrib) {
			px = pxad;
			pygx = pygxad;
		} else {
			px = pxbd;
			pygx = pygxbd;
		}

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		int[] indexes = new int[nAttributes.getValue()];

		/*System.out.println("Setting Values for x_n");*/
		// setting values for x_1,...,x_n
		for (int a = 0; a < indexes.length; a++) {
			/*System.out.println("a: " + a);*/
			// choosing values of x_1,...,x_n
			double rand = r.nextUniform(0.0, 1.0, true);
			int chosenVal = 0;
			double sumProba = px[a][chosenVal];
			while (rand > sumProba) {
				/*System.out.println("class val: " + chosenVal);*/
				chosenVal++;
				sumProba += px[a][chosenVal];
			}
			indexes[a] = chosenVal;
			inst.setValue(a, chosenVal);
		}

		int lineNoCPT = getIndex(indexes);

		/*System.out.println("Setting Class Values");*/
		int chosenClassValue = 0;
		while (pygx[lineNoCPT][chosenClassValue] != 1.0) {
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

		pxbd = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
		pygxbd = new double[nCombinationsValuesForPX][nValuesPerAttribute.getValue()];

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);

		// generating distribution before drift

		// p(x)
		generateRandomPx(pxbd, r);

		// p(y|x)
		generateRandomPyGivenX(pygxbd, r);

		// generating distribution after drift

		if (driftPriors.isSet()) {
			pxad = new double[nAttributes.getValue()][nValuesPerAttribute.getValue()];
			
			double obtainedMagnitude;
			System.out.println("Sampling p(x) for required magnitude...");
			do {
				if (driftMagnitudePrior.getValue() >= 0.2) {
					generateRandomPx(pxad, r);
				} else if (driftMagnitudePrior.getValue() < 0.2) {
					generateRandomPxAfterCloseToBefore(driftMagnitudePrior.getValue(), pxbd, pxad, r);
				}
				obtainedMagnitude = computeMagnitudePX(nCombinationsValuesForPX, pxbd, pxad);
			} while (Math.abs(obtainedMagnitude - driftMagnitudePrior.getValue()) > precisionDriftMagnitude.getValue());

			System.out.println("exact magnitude for p(x)=" + computeMagnitudePX(nCombinationsValuesForPX, pxbd, pxad) + "\tasked=" + driftMagnitudePrior.getValue());
		} else {
			pxad = pxbd;
		}

		// conditional
		if (driftConditional.isSet()) {
			pygxad = new double[nCombinationsValuesForPX][];
			for (int line = 0; line < pygxad.length; line++) {
				// default is same distrib
				pygxad[line] = pygxbd[line];
			}

			int nLinesToChange = (int) Math.round(driftMagnitudeConditional.getValue() * nCombinationsValuesForPX);
			if (nLinesToChange == 0.0) {
				System.out.println("Not enough drift to be noticeable in p(y|x) - unchanged");
				pygxad = pygxbd;
			} else {
				int[] linesToChange = r.nextPermutation(nCombinationsValuesForPX, nLinesToChange);

				for (int line : linesToChange) {
					pygxad[line] = new double[nValuesPerAttribute.getValue()];

					double[] lineCPT = pygxad[line];
					int chosenClass;

					do {
						chosenClass = r.nextInt(0, lineCPT.length - 1);
						// making sure we choose a different class value
					} while (pygxbd[line][chosenClass] == 1.0);

					for (int c = 0; c < lineCPT.length; c++) {
						if (c == chosenClass) {
							lineCPT[c] = 1.0;
						} else {
							lineCPT[c] = 0.0;
						}
					}
				}

				System.out.println("exact magnitude for p(y|x)=" + computeMagnitudePYGX(pygxbd, pygxad) + "\tasked=" + driftMagnitudeConditional.getValue());
			}
		} else {
			pygxad = pygxbd;
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
