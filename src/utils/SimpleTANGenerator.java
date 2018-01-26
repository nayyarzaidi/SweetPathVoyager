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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

public class SimpleTANGenerator extends DriftGenerator {

	private static final long serialVersionUID = 1291115908166720203L;

	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000);
	public IntOption frequency = new IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000);

	protected InstancesHeader streamHeader;

	double[] p_y;
	double[][] p_yx;
	double[][][][] p_yxx;

	int[] parents;

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

		//FastVector<Attribute> attributes = getHeaderAttributes(nAttributes.getValue(), nValuesPerAttribute.getValue());

		int n = nAttributes.getValue() ;
		int nvals = nValuesPerAttribute.getValue();

		FastVector<Attribute> attributes = new FastVector<>();
		List<String> attributeValues = new ArrayList<String>();
		for (int v = 0; v < nvals; v++) {
			attributeValues.add("v" + (v + 1));
		}
		for (int i = 0; i < n; i++) {
			attributes.addElement(new Attribute("x" + (i + 1), attributeValues));
		}
		List<String> classValues = new ArrayList<String>();

		for (int v = 0; v < Globals.getNumClasses(); v++) {
			classValues.add("class" + (v + 1));
		}
		attributes.addElement(new Attribute("class", classValues));

		this.streamHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
	}

	@Override
	public InstanceExample nextInstance() {

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		int nc = Globals.getNumClasses();
		int n = streamHeader.numAttributes() - 1;

		int y = r.nextInt(0, nc - 1);
		inst.setClassValue(y);

		int[] x = new int[n];

		x[0] = SUtils.sampleFromNonUniformDistribution(p_yx[y], r);

		inst.setValue(0, x[0]);

		for (int i = 1; i < n;  i++) {
			int p = parents[i];
			int xp = x[p];  

			//System.out.println("Parent of " + i + " is " + parents[i] + " which takes value of " + xp);

			x[i] = SUtils.sampleFromNonUniformDistribution(p_yxx[i][y][xp], r);

			inst.setValue(i, x[i]);
		}

		nInstancesGeneratedSoFar++;

		return new InstanceExample(inst);
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

		generateHeader();

		int nc = Globals.getNumClasses();
		int n = streamHeader.numAttributes() - 1;
		int nvals = nValuesPerAttribute.getValue();

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);

		parents = new int[n];
		parents[1] = 0;
		for (int i = 2; i < n; i++) {
			int p = r.nextInt(0, i-1);
			parents[i] = p;
		}
		//		parents[1] = 0;
		//		for (int i = 2; i < n; i++) {
		//			parents[i] = i - 1;
		//		}	
		System.out.println("Seed = " + seed.getValue());
		System.out.println(Arrays.toString(parents));

		/* Declaration */

		p_y = new double[nc];

		p_yx = new double[nc][];
		for (int y = 0; y < nc; y++) {
			p_yx[y] = new double[nvals];
		}

		p_yxx = new double[n][][][];

		for (int i = 1; i < n; i++) {
			p_yxx[i] = new double[nc][][];
			for (int y = 0; y < nc; y++) {
				p_yxx[i][y] = new double[nvals][];
				for (int x = 0; x < nvals; x++) {
					p_yxx[i][y][x] = new double[nvals];
				}
			}
		}

		/* Initialization */

		double sum = 0;
		for (int y = 0; y < nc; y++) {
			p_y[y] = r.nextUniform(0, 1);
			sum += p_y[y];
		}
		for (int y = 0; y < nc; y++) {
			p_y[y] /= sum;

		}

		//for (int y = 0; y < nc; y++) {
		//	p_y[y] = (double)1/nc;
		//}

		for (int y = 0; y < nc; y++) {

			sum = 0;
			for (int x1 = 0; x1 < nvals; x1++) {
				//p_yx[y][x1] = (double)1/nvals;
				p_yx[y][x1] = r.nextUniform(0, 1);

				sum += p_yx[y][x1];
			}

			for (int x1 = 0; x1 < nvals; x1++) {
				p_yx[y][x1] /= sum;
			}

		}

		for (int i = 1; i < n; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x1 = 0; x1 < nvals; x1++) {

					sum = 0;
					for (int x2 = 0; x2 < nvals; x2++) {
						//p_yxx[i][y][x1][x2] = (double)1/nvals;
						p_yxx[i][y][x1][x2] = r.nextUniform(0, 1);

						sum += p_yxx[i][y][x1][x2];
					}

					for (int x2 = 0; x2 < nvals; x2++) {
						p_yxx[i][y][x1][x2] /= sum;
					}

				}
			}
		}

		nInstancesGeneratedSoFar = 0L;
	}

}
