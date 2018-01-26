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

public class SimpleDriftGenerator extends DriftGenerator {

	private static final long serialVersionUID = 1291115908166720203L;

	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000);
	public IntOption frequency = new IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000);

	protected InstancesHeader streamHeader;

	double[][] pxbd;
	double[][] pygxbd;

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

//		if ((nInstancesGeneratedSoFar % frequency.getValue()) == 0) {			
//			generateRandomPyGivenX(pygxbd, r);
//		}

		//		if ((nInstancesGeneratedSoFar % frequency.getValue()) == 0) {
		//			// Change pygxbd
		//			int lineNoCPT = r.nextInt(0, pygxbd.length - 1);
		//
		//			double[] line = pygxbd[lineNoCPT];
		//			//System.out.print("Changing tuple:" + lineNoCPT + " - Before: " + Arrays.toString(line));
		//			
		//			int chosen = 0;
		//			do {
		//				chosen = r.nextInt(0, line.length - 1);
		//			} while(line[chosen] == 1);
		//
		//			for (int i = 0; i < line.length; i++) {
		//				pygxbd[lineNoCPT][i] = 0;
		//			}
		//			pygxbd[lineNoCPT][chosen] = 1;
		//			
		//			//System.out.print(", After: " + Arrays.toString(pygxbd[lineNoCPT]) + "\n");
		//		}

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
			double sumProba = pxbd[a][chosenVal];
			while (rand > sumProba) {
				/*System.out.println("class val: " + chosenVal);*/
				chosenVal++;
				sumProba += pxbd[a][chosenVal];
			}
			indexes[a] = chosenVal;
			inst.setValue(a, chosenVal);
		}

		int lineNoCPT = getIndex(indexes);

		/*System.out.println("Setting Class Values");*/
//		int chosenClassValue = 0;
//		while (pygxbd[lineNoCPT][chosenClassValue] != 1.0) {
//			chosenClassValue++;
//		}
//		inst.setClassValue(chosenClassValue);
		
		double rand = r.nextUniform(0.0, 1.0, true);
		int chosenClassValue = 0;
		double sumProba = pygxbd[lineNoCPT][chosenClassValue];
		while (rand > sumProba) {
			chosenClassValue++;
			sumProba += pygxbd[lineNoCPT][chosenClassValue];
		}
		inst.setClassValue(chosenClassValue);
		

		nInstancesGeneratedSoFar++;

		return new InstanceExample(inst);
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

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

		// p(x)
		//generateRandomPx(pxbd, r);
		generateUniformPx(pxbd);

		// p(y|x)
		generateRandomPyGivenX(pygxbd, r);
		//generateRandomPyGivenX(pygxbd, r, 0.3);
		
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
