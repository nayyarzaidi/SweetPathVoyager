package utils;

/*
 * Created by Lee on 6/09/2015.
 * Randomly change drift direction
 */

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;
import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SwitchingKDBGeneratorWithDriftBinaryValues extends DriftGenerator {

    private static final long serialVersionUID = 1291115908166720203L;

	public IntOption driftLength = new IntOption("driftLength", 'l', "The number of instances the concept drift will be applied acrros", 0, 0, 10000000);
	public IntOption frequency = new IntOption("frequency", 'F', "The number of instances after which to change an entry in pygxbd", 0, 0, 100000);
    public FloatOption dirChangeProb = new FloatOption(
            "dirChangeProb",
            'd',
            "The probability that the drift of an entry in the CPTs will change",
            0.02, 0.0, 1.0);

	protected InstancesHeader streamHeader;

	double[] p_y;
	double[][][] p_yx;
	double[][][][] p_yxx;
	double[][][][][] p_yxxx;

	int L1 = 0;
	int L2 = 0;
	int L3 = 0;
	int L4 = 0;

	int d_yx[][];  // direction for yx - initialise randomly to 0 or 1
	int d_yxx[][][];  // direction for yxx - initialise randomly to 0 or 1
	int d_yxxx[][][][]; // direction for yxxx - initialise randomly to 0 or 1

	int[] randAttributes;

	RandomDataGenerator r;

	long nInstancesGeneratedSoFar;

    private double delta = Globals.getDriftDelta();

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
		int nvals = 2;

		//		L1 = 0;
		//		L2 = n/3;
		//		L3 = (2 * n)/3;
		//		L4 = n;

		//L1 = 0;
		//L2 = 1;
		//L3 = 2;
		//L4 = n;

		L1 = 0;
		L2 = 2;
		L3 = 50;
		L4 = n;

		System.out.println("Out of " + n + " attributes, " + ((L2 - L1)) + " of them will be order 1 and " + ((L3 - L2))  + " will be order 2, and " + ((L4 - 1 - L3)) + " will be order 3.");

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

        int n = streamHeader.numAttributes() - 1;

		/* Start: Introduce some Posterior-drift before generating the data */

        int numDriftAttributes = (int)Globals.getDriftMagnitude();

        if (numDriftAttributes > n) {
            System.out.println("Number of attributes with drift can't be greater than actual number of attributes");
            System.exit(-1);
        }

		randAttributes = new int[numDriftAttributes];
		for (int i = 0; i < randAttributes.length; i++) {
			randAttributes[i] = -1;
		}

		int size = 0;
		while (size < numDriftAttributes) {
			int p = r.nextInt(2, n-1);

			if (!SUtils.inArray(p, randAttributes)) {
				randAttributes[size] = p;
				size++;
			}
		}

        //System.out.println("Rand Attributes to be drifted are " + Arrays.toString(randAttributes));

        for (int i = 0; i < numDriftAttributes; i++) {
            int p = randAttributes[i];

            if (p >= L1 && p < L2) {

                intoducePosteriorDrift(p_yx[p], d_yx[p], delta);

            } else if (p >= L2 && p < L3) {

                intoducePosteriorDrift(p_yxx[p], d_yxx[p], delta);

            } else if (p >= L3 && p < L4) {

                intoducePosteriorDrift(p_yxxx[p], d_yxxx[p], delta);

            }

        }

		/* End: Introduce some Posterior-drift before generating the data */

        Instance inst = new DenseInstance(streamHeader.numAttributes());
        inst.setDataset(streamHeader);

        int nc = Globals.getNumClasses();

        int y = r.nextInt(0, nc - 1);
        inst.setClassValue(y);

        int[] x = new int[n];
        Arrays.fill(x, -1);

        x[0] = SUtils.sampleFromNonUniformDistribution(p_yx[0][y], r);

        inst.setValue(0, x[0]);

        x[1] = SUtils.sampleFromNonUniformDistribution(p_yxx[1][y][x[0]], r);

        inst.setValue(1, x[1]);

        for (int i = 2; i < n;  i++) {

            if (i >= L1 && i < L2) {

                x[i] = SUtils.sampleFromNonUniformDistribution(p_yx[i][y], r);

            } else if (i >= L2 && i < L3) {

                int p1 = 0;
                int xp1 = x[p1];

                x[i] = SUtils.sampleFromNonUniformDistribution(p_yxx[i][y][xp1], r);

            } else if (i >= L3 && i < L4) {

                int p1 =  r.nextInt(0, 1);
                int p2 = 1 - p1;

                int xp1 = x[p1];
                int xp2 = x[p2];

                x[i] = SUtils.sampleFromNonUniformDistribution(p_yxxx[i][y][xp1][xp2], r);

            }

            inst.setValue(i, x[i]);
        }

        nInstancesGeneratedSoFar++;

        return new InstanceExample(inst);
    }

    private int intoducePosteriorDrift(double[] probDist, int direction, double delta) {
        int currentDir = direction == 1 ? 1 : -1;
        // Case to change direction
        if (probDist[0] < delta) {
            currentDir = 1;
        }
        else if (probDist[0] > 1.0 - delta) {
            currentDir = -1;
        }
        else if (r.nextUniform(0,1) < dirChangeProb.getValue()) {
            currentDir *= -1;
        }
        probDist[0] += (currentDir * delta);
        probDist[1] = 1.0 - probDist[0];
        return currentDir == 1 ? 1 : 0;
    }

	private void intoducePosteriorDrift(double p_yx[][], int d_yx[], double delta) {


		int y;  // the class

		for (y = 0; y <= 1; y++) {
		    d_yx[y] = intoducePosteriorDrift(p_yx[y], d_yx[y], delta);
		}
	}

	private void intoducePosteriorDrift(double p_yxx[][][], int d_yxx[][], double delta) {

		int p;  // the parent
		int y;  // the class

		for (p = 0; p <= 1; p++) {
			for (y = 0; y <= 1; y++) {
                d_yxx[y][p] = intoducePosteriorDrift(p_yxx[y][p], d_yxx[y][p], delta);
			}
		}

	}

	private void intoducePosteriorDrift(double p_yxxx[][][][], int d_yxxx[][][], double delta) {

		int p1;  // parent 1
		int p2; // parent 2
		int y;  // the class

		for (p1 = 0; p1 <= 1; p1++) {
			for (p2 = 0; p2 <= 1; p2++) {
				for (y = 0; y <= 1; y++) {
                    d_yxxx[y][p1][p2] = intoducePosteriorDrift(p_yxxx[y][p1][p2], d_yxxx[y][p1][p2], delta);
				}
			}
		}

	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);

		generateHeader();

		int nc = Globals.getNumClasses();
		int n = streamHeader.numAttributes() - 1;
		int nvals = 2;

		System.out.println("Seed = " + seed.getValue());

		/*
		int numDriftAttributes = (int)Globals.getDriftMagnitude();

        if (numDriftAttributes > n) {
            System.out.println("Number of attributes with drift can't be greater than actual number of attributes");
            System.exit(-1);
        }
		randAttributes = new int[numDriftAttributes];
        for (int i = 0; i < randAttributes.length; i++) {
            randAttributes[i] = -1;
        }
        */

		/* Declaration */

		p_y = new double[nc];

		p_yx = new double[n][][];

		p_yx[0] = new double[nc][];
		for (int y = 0; y < nc; y++) {
			p_yx[0][y] = new double[nvals];
		}

		for (int i = L1; i < L2; i++) {
			p_yx[i] = new double[nc][];
			for (int y = 0; y < nc; y++) {
				p_yx[i][y] = new double[nvals];
			}
		}

		p_yxx = new double[n][][][];

		p_yxx[1] = new double[nc][][];
		for (int y = 0; y < nc; y++) {
			p_yxx[1][y] = new double[nvals][];
			for (int x1 = 0; x1 < nvals; x1++) {
				p_yxx[1][y][x1] = new double[nvals];
			}
		}

		for (int i = L2; i < L3; i++) {
			p_yxx[i] = new double[nc][][];
			for (int y = 0; y < nc; y++) {
				p_yxx[i][y] = new double[nvals][];
				for (int x1 = 0; x1 < nvals; x1++) {
					p_yxx[i][y][x1] = new double[nvals];
				}
			}
		}

		p_yxxx = new double[n][][][][];

		for (int i = L3; i < L4; i++) {
			p_yxxx[i] = new double[nc][][][];
			for (int y = 0; y < nc; y++) {
				p_yxxx[i][y] = new double[nvals][][];
				for (int x1 = 0; x1 < nvals; x1++) {
					p_yxxx[i][y][x1] = new double[nvals][];
					for (int x2 = 0; x2 < nvals; x2++) {
						p_yxxx[i][y][x1][x2] = new double[nvals];
					}
				}
			}
		}


		d_yx = new int[n][];

		for (int i = L1; i < L2; i++) {
			d_yx[i] = new int[nc];
		}

		d_yxx = new int[n][][];

		for (int i = L2; i < L3; i++) {
			d_yxx[i] = new int[nc][];
			for (int y = 0; y < nc; y++) {
				d_yxx[i][y] = new int[nvals];
			}
		}

		d_yxxx = new int[n][][][];

		for (int i = L3; i < L4; i++) {
			d_yxxx[i] = new int[nc][][];
			for (int y = 0; y < nc; y++) {
				d_yxxx[i][y] = new int[nvals][];
				for (int x1 = 0; x1 < nvals; x1++) {
					d_yxxx[i][y][x1] = new int[nvals];
				}
			}
		}

		/* Initialization */
		intializeCPTBinary();

		nInstancesGeneratedSoFar = 0L;
	}

	public void intializeCPTBinary() {

		int nc = Globals.getNumClasses();
		int n = streamHeader.numAttributes() - 1;
		int nvals = 2;

		p_y[0] = r.nextUniform(0, 1);
		p_y[1] = 1.0 - p_y[0];

		for (int y = 0; y < nc; y++) {
			p_yx[0][y][0] = r.nextUniform(0, 1);
			p_yx[0][y][1] = 1.0 - p_yx[0][y][0];
		}

		for (int i = L1; i < L2; i++) {
			for (int y = 0; y < nc; y++) {
				p_yx[i][y][0] = r.nextUniform(0, 1);
				p_yx[i][y][1] = 1.0 - p_yx[i][y][0];
			}
		}

		for (int y = 0; y < nc; y++) {
			for (int x1 = 0; x1 < nvals; x1++) {
				p_yxx[1][y][x1][0] = r.nextUniform(0, 1);
				p_yxx[1][y][x1][1] = 1 - p_yxx[1][y][x1][0];
			}
		}

		for (int i = L2; i < L3; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x1 = 0; x1 < nvals; x1++) {
					p_yxx[i][y][x1][0] = r.nextUniform(0, 1);
					p_yxx[i][y][x1][1] = 1 - p_yxx[i][y][x1][0];
				}
			}
		}

		for (int i = L3; i < L4; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x1 = 0; x1 < nvals; x1++) {
					for (int x2 = 0; x2 < nvals; x2++) {
						p_yxxx[i][y][x1][x2][0] = r.nextUniform(0, 1);
						p_yxxx[i][y][x1][x2][1] = 1 - p_yxxx[i][y][x1][x2][0];
					}
				}
			}
		}

		for (int i = L1; i < L2; i++) {
			for (int y = 0; y < nc; y++) {
				d_yx[i][y] = r.nextInt(0, 1);
				d_yx[i][y] = r.nextInt(0, 1);
			}
		}

		for (int i = L2; i < L3; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x = 0; x < nvals; x++) {
					d_yxx[i][y][x] = r.nextInt(0, 1);
					d_yxx[i][y][x] = r.nextInt(0, 1);
				}
			}
		}

		for (int i = L3; i < L4; i++) {
			for (int y = 0; y < nc; y++) {
				for (int x1 = 0; x1 < nvals; x1++) {
					for (int x2 = 0; x2 < nvals; x2++) {
						d_yxxx[i][y][x1][x2] = r.nextInt(0, 1);
						d_yxxx[i][y][x1][x2] = r.nextInt(0, 1);
					}
				}
			}
		}

	}

}
