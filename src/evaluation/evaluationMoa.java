package evaluation;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.math3.random.MersenneTwister;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.driftdetection.DDM;
import moa.classifiers.core.driftdetection.EDDM;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.classifiers.functions.SGDMultiClass;
import moa.classifiers.meta.AccuracyUpdatedEnsemble;
import moa.classifiers.meta.AccuracyWeightedEnsemble;
import moa.classifiers.meta.LeveragingBag;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.meta.OzaBagAdwin;
import moa.classifiers.meta.OzaBoost;
import moa.classifiers.meta.OzaBoostAdwin;
import moa.classifiers.trees.ASHoeffdingTree;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ArffFileStream;
import model.Model;
import model.ande.ande;
import utils.Globals;
import utils.SUtils;
import utils.filters.ida.IDADiscretizer;
import utils.filters.ida.IDADiscretizer.IDAType;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class evaluationMoa {

	static DecimalFormat df = new DecimalFormat("#.####");

	public static void learn() throws Exception {

		Instances structure = null;

		String option = "AnDEExperiments";

		if (option.equalsIgnoreCase("AnDEExperiments")) {

			String[] datasets = new String[]{"powersupply.arff", "elecNormNew.arff", "airlines.arff", "sensor.arff"};

			for (int d = 0; d < datasets.length; d++) {
				String dataName = datasets[d];
		
				//File sourceFile = new File(Globals.getDatasetRepository() + dataName);
				//Globals.setSOURCEFILE(sourceFile);
				//structure = SUtils.setStructure();

				double[] decays = {0.5, 0.1, 0.01, 0.001, 0.0001};
				int numDecays = decays.length;

				double[] windows = {10, 20, 50, 100, 500, 1000};
				int numWindows = windows.length;

				Model[] learnersDecayNB = new Model[numDecays];
				Model[] learnersDecayA1DE = new Model[numDecays];
				Model[] learnersDecayA2DE = new Model[numDecays];

				double[][][] resDecay = new double[datasets.length][decays.length][3];

				Model[] learnersWindowNB = new Model[numWindows];
				Model[] learnersWindowA1DE = new Model[numWindows];
				Model[] learnersWindowA2DE = new Model[numWindows];

				for (int i = 0; i < numDecays; i++) {
					
					ArffFileStream stream = new ArffFileStream(Globals.getDatasetRepository() + dataName, -1);
					stream.prepareForUse();

					IDADiscretizer filter = new IDADiscretizer(5, 1000, IDAType.IDAW);
					filter.setInputStream(stream);
					filter.init();
					filter.prepareForUse();

					Globals.setAdaptiveControl("decay");
					Globals.setAdaptiveControlParameter(decays[i]);

					int level = 0;
					Globals.setLevel(level);
					learnersDecayNB[i] = new ande();
					double error = evaluateLearner(learnersDecayNB[i], filter);
					resDecay[d][i][level] = error;

					level = 1;
					Globals.setLevel(level);
					learnersDecayA1DE[i] = new ande();
					error = evaluateLearner(learnersDecayA1DE[i], filter);
					resDecay[d][i][level] = error;

					level = 2;
					Globals.setLevel(level);
					learnersDecayA2DE[i] = new ande();
					error = evaluateLearner(learnersDecayA2DE[i], filter);
					resDecay[d][i][level] = error;
				}

				for (int i = 0; i < numDecays; i++) {
					
					ArffFileStream stream = new ArffFileStream(Globals.getDatasetRepository() + dataName, -1);
					stream.prepareForUse();

					IDADiscretizer filter = new IDADiscretizer(5, 1000, IDAType.IDAW);
					filter.setInputStream(stream);
					filter.init();
					filter.prepareForUse();
					
					Globals.setAdaptiveControl("window");
					Globals.setAdaptiveControlParameter(windows[i]);

					int level = 0;
					Globals.setLevel(level);
					learnersWindowNB[i] = new ande();
					double error = evaluateLearner(learnersWindowNB[i], filter);
					resDecay[d][i][level] = error;

					level = 1;
					Globals.setLevel(level);
					learnersWindowA1DE[i] = new ande();
					error = evaluateLearner(learnersWindowA1DE[i], filter);
					resDecay[d][i][level] = error;

					level = 2;
					Globals.setLevel(level);
					learnersWindowA2DE[i] = new ande();
					error = evaluateLearner(learnersWindowA2DE[i], filter);
					resDecay[d][i][level] = error;
				}

			}

		} else if (option.equalsIgnoreCase("MOAExperiments")) {

			/* 
			 * Test MOA Classifiers
			 */

			String[] datasets = new String[]{"powersupply.arff", "elecNormNew.arff", "airlines.arff", "sensor.arff"};

			for (String string : datasets) {
				String dataName = string.substring(0, string.lastIndexOf("."));

				//				File f = new File(Globals.getOuputResultsDirectory() + "/" + dataName + "_results.csv");

				//				if (f.exists()) {
				//					System.out.println("Already treated or ongoing for " + string);
				//					continue;
				//				} else {
				//					f.createNewFile();
				//				}

				PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(Globals.getOuputResultsDirectory() + "/" + dataName + "_results.csv", false)));

				ArrayList<Classifier> learners = new ArrayList<>();

				//				learners.add(new AccuracyUpdatedEnsemble());
				//				learners.add(new OzaBagAdwin());

				//				DriftDetectionMethodClassifier ddm = new DriftDetectionMethodClassifier();
				//				ddm.driftDetectionMethodOption.setCurrentObject(new DDM());
				//				learners.add(ddm);
				//
				//				DriftDetectionMethodClassifier eddm = new DriftDetectionMethodClassifier();
				//				eddm.driftDetectionMethodOption.setCurrentObject(new EDDM());
				//				learners.add(eddm);

				//				learners.add(new ASHoeffdingTree());
				//				learners.add(new HoeffdingTree());
				//				learners.add(new OzaBag());
				//				learners.add(new NaiveBayes());
				//				learners.add(new HoeffdingAdaptiveTree());
				//				learners.add(new OzaBoost());
				//				learners.add(new AccuracyWeightedEnsemble());
				//				learners.add(new LeveragingBag());
				//				learners.add(new OzaBoostAdwin());
				//
				//				SGDMultiClass lrsgd = new SGDMultiClass();
				//				lrsgd.setLambda(0.001);
				//				lrsgd.setLearningRate(0.001);
				//				lrsgd.setLossFunction(1);
				//				learners.add(new SGDMultiClass());

				// get the dataset
				ArffFileStream stream = new ArffFileStream(Globals.getTempDirectory() + string, -1);
				stream.prepareForUse();

				// get the discretizer
				IDADiscretizer filter = new IDADiscretizer(5,1000,IDAType.IDAW);
				filter.setInputStream(stream);
				filter.init();
				filter.prepareForUse();

				// prepare the models
				for (Classifier classifier : learners) {
					classifier.setModelContext(filter.getHeader());
					classifier.prepareForUse();
				}

				double[] nErrors = new double[learners.size()];
				int numberSamples = 0;

				while (filter.hasMoreInstances()) {
					Instance trainInst = (Instance) filter.nextInstance().getData();

					// try to classify the instance with each classifier
					for (int i = 0; i < learners.size(); i++) {
						Classifier cls = learners.get(i);
						if(!cls.correctlyClassifies(trainInst)) {
							nErrors[i]++;
						}
					}

					numberSamples++;

					// train each classifier with the instance
					for (int i = 0; i < learners.size(); i++) {
						learners.get(i).trainOnInstance(trainInst);
					}
				}

				// output the results
				out.println(string);
				for (int i = 0; i < learners.size(); i++) {
					Classifier classifier = learners.get(i);
					String fName = classifier.getClass().getSimpleName();
					out.println(fName+"\t"+(df.format(1.0 * nErrors[i]/numberSamples)));
				}
				out.println();
				out.flush();
			}

		}

	}

	private static double evaluateLearner(Model learner, IDADiscretizer filter) throws FileNotFoundException, IOException {
		double error = 0;
		
		int nErrors = 0;
		int lineNo = 0;

		while (filter.hasMoreInstances()) {

			Instance row = (Instance) filter.nextInstance().getData();
			
			double[] probs = learner.distributionForInstance((weka.core.Instance) row);

			double[] results = SUtils.getResults(probs, (int) row.classValue(), probs.length);

			if (results[1] == 1) {
				nErrors++;
			}

			learner.update((weka.core.Instance) row);

			lineNo++;
		}

		error = nErrors/lineNo;

		return error;
	} 

}
