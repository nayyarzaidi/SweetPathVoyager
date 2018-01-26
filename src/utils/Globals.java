package utils;

import java.io.File;
import java.util.Map;

import weka.core.Instances;

public class Globals {
	
	private static String dataSetName = "";
	
	private static int BUFFER_SIZE  = 10*1024*1024; 	//100MB
	private static int ARFF_BUFFER_SIZE = 100000;
	
	private static File SOURCEFILE;
	private static File CVFILE;
	
	private static boolean cvFilePresent = false;
	
	private static String version = "v0.2";
	
	private static boolean verbose = false;
	
	private static String experimentType  = "";
	
	private static String trainFile = "";
	private static String testFile = "";
	private static String cvFile = "";
	
	private static int numExp  = 5;
	private static int numFolds = 2;

	private static String model = "AnDE";
	
	private static int level = 0;
	
	private static boolean doWanbiac = false;
	private static boolean doDiscriminative = false;

	private static boolean doSelectiveKDB = false;
	private static boolean doRegularization = false;
	private static double  lambda  = 0.01;
	private static String adaptiveRegularization = "None";
	
	private static boolean doBinaryClassification = false;
	
	private static String objectiveFunction = "CLL";
	
	private static String dataStructureParameter = "Flat";
	
	private static int hashParameter  = 60000000;

	private static String regularizationType = "L2";
	private static double regularizationTowards = 0;

	private static String optimizer = "sgd";
	private static int numIterations  = 10;
	
	private static String sgdType = "Adagrad";
	private static double sgdTuningParameter = 0.01;
	private static boolean doCrossValidateTuningParameter  = false;

	private static String discretization = "None";
	private static boolean normalizeNumeric = false;

	private static int discretizationParameter  = 10;
	
	private static int holdoutsetPrecentage = 5;
	
	private static String mcmcType = "ALS";
	
	private static String featureSelection = "None";
	private static double featureSelectionParameter = 1.0;
	
	private static double numberInstances = 0;
	private static int numClasses = 2;
	private static int numAttributes = 0;

	private static int numRandAttributes = 3;
	
	private static int[] paramsPerAtt = null;
	private static boolean[] isNumericTrue = null;
	
	private static Instances CVInstances = null;

	private static Instances structure = null;
	
	private static boolean numInstancesKnown = false;
	
	private static String adaptiveControl  = "None";
	private static double adaptiveControlParameter = 0.1;
	
	private static int prequentialOutputResolution = 1;
	
	private static boolean doMovingAverage = false;

	private static int prequentialBufferOutputResolution = 100;
	
	private static boolean computeProbabilitiesFromCount = false;
	
	private static String tempDirectory = "/Users/nayyar/Desktop/AA/temp";

	private static String ouputResultsDirectory = "/Users/nayyar/Desktop/AA/output";
	
	private static String datasetRepository = "/Users/nayyar/WData/datasets_Decay/";
	
	private static String flowParameter = "adaptiveControlParameter:{0,0.1,0.01,0.001}";
	
	private static String driftType = "noDrift";
	
	private static boolean generateDriftData = false;
	
	//private static int totalNInstancesBeforeDrift = 100000;
	private static int totalNInstancesBeforeDrift = 0;
	
	private static int totalNInstancesDuringDrift = 10000;

	//private static int totalNInstancesAfterDrift = 100000;
	private static int totalNInstancesAfterDrift = 0;
	
	private static int driftNAttributes = 5;
	
	private static int driftNAttributesValues = 3;
	
	private static double driftMagnitude = 0.4;
	
	private static int driftMagnitude2 = 10;
	
	private static double driftMagnitude3 = 0.1;
	
	private static boolean discretizeOutOfCore = false;
	
	private static String preProcessParameter = "discretize";
	
	private static String ignoreAttributes = "";
	
	private static int classAttribute = -1;
	
	private static String attributeType = "";
	
	private static int dicedPercentage = 10;
	
	private static boolean dicedStratified = false;
	
	private static int dicedAt = 0;
	
	private static Map<String, Double> fsScore = null;
	
	private static int latentK = 10;
	
	private static double driftDelta = 0.1;
	
	private static boolean plotRMSEResuts = false;
			
	public static String getVersion() {
		return version;
	}

	public static void setVersion(String version) {
		Globals.version = version;
	}

	public static boolean isVerbose() {
		return verbose;
	}

	public static void setVerbose(boolean verbose) {
		Globals.verbose = verbose;
	}

	public static String getExperimentType() {
		return experimentType;
	}

	public static void setExperimentType(String experimentType) {
		Globals.experimentType = experimentType;
	}

	public static String getTrainFile() {
		return trainFile;
	}

	public static void setTrainFile(String trainFile) {
		Globals.trainFile = trainFile;
	}

	public static String getTestFile() {
		return testFile;
	}

	public static void setTestFile(String testFile) {
		Globals.testFile = testFile;
	}

	public static int getNumExp() {
		return numExp;
	}

	public static void setNumExp(int numExp) {
		Globals.numExp = numExp;
	}

	public static int getNumFolds() {
		return numFolds;
	}

	public static void setNumFolds(int numFolds) {
		Globals.numFolds = numFolds;
	}

	public static String getModel() {
		return model;
	}

	public static void setModel(String model) {
		Globals.model = model;
	}

	public static int getLevel() {
		return level;
	}

	public static void setLevel(int level) {
		Globals.level = level;
	}

	public static boolean isDoWanbiac() {
		return doWanbiac;
	}

	public static void setDoWanbiac(boolean doWanbiac) {
		Globals.doWanbiac = doWanbiac;
	}

	public static boolean isDoSelectiveKDB() {
		return doSelectiveKDB;
	}

	public static void setDoSelectiveKDB(boolean doSelectiveKDB) {
		Globals.doSelectiveKDB = doSelectiveKDB;
	}

	public static boolean isDoRegularization() {
		return doRegularization;
	}

	public static void setDoRegularization(boolean doRegularization) {
		Globals.doRegularization = doRegularization;
	}

	public static double getLambda() {
		return lambda;
	}

	public static void setLambda(double lambda) {
		Globals.lambda = lambda;
	}

	public static String getAdaptiveRegularization() {
		return adaptiveRegularization;
	}

	public static void setAdaptiveRegularization(String adaptiveRegularization) {
		Globals.adaptiveRegularization = adaptiveRegularization;
	}

	public static String getOptimizer() {
		return optimizer;
	}

	public static void setOptimizer(String optimizer) {
		Globals.optimizer = optimizer;
	}

	public static int getNumIterations() {
		return numIterations;
	}

	public static void setNumIterations(int numIterations) {
		Globals.numIterations = numIterations;
	}

	public static String getSgdType() {
		return sgdType;
	}

	public static void setSgdType(String sgdType) {
		Globals.sgdType = sgdType;
	}

	public static double getSgdTuningParameter() {
		return sgdTuningParameter;
	}

	public static void setSgdTuningParameter(double sgdTuningParameter) {
		Globals.sgdTuningParameter = sgdTuningParameter;
	}

	public static boolean isDoCrossValidateTuningParameter() {
		return doCrossValidateTuningParameter;
	}

	public static void setDoCrossValidateTuningParameter(boolean doCrossValidateTuningParameter) {
		Globals.doCrossValidateTuningParameter = doCrossValidateTuningParameter;
	}
	
	public static String getRegularizationType() {
		return regularizationType;
	}

	public static void setRegularizationType(String regularizationType) {
		Globals.regularizationType = regularizationType;
	}

	public static double getRegularizationTowards() {
		return regularizationTowards;
	}

	public static void setRegularizationTowards(double regularizationTowards) {
		Globals.regularizationTowards = regularizationTowards;
	}

	public static String getMcmcType() {
		return mcmcType;
	}

	public static void setMcmcType(String mcmcType) {
		Globals.mcmcType = mcmcType;
	}

	public static void setDiscretization(String discretization) {
		Globals.discretization = discretization;
	}

	public static boolean isNormalizeNumeric() {
		return normalizeNumeric;
	}

	public static void setNormalizeNumeric(boolean normalizeNumeric) {
		Globals.normalizeNumeric = normalizeNumeric;
	}

	public static int getDiscretizationParameter() {
		return discretizationParameter;
	}
	
	public static String getDiscretization() {
		return discretization;
	}

	public static void setDiscretizationParameter(int discretizationParameter) {
		Globals.discretizationParameter = discretizationParameter;
	}
	
	public static int getBUFFER_SIZE() {
		return BUFFER_SIZE;
	}

	public static void setBUFFER_SIZE(int bUFFER_SIZE) {
		BUFFER_SIZE = bUFFER_SIZE;
	}

	public static int getARFF_BUFFER_SIZE() {
		return ARFF_BUFFER_SIZE;
	}

	public static void setARFF_BUFFER_SIZE(int aRFF_BUFFER_SIZE) {
		ARFF_BUFFER_SIZE = aRFF_BUFFER_SIZE;
	}

	public static File getSOURCEFILE() {
		return SOURCEFILE;
	}

	public static void setSOURCEFILE(File sOURCEFILE) {
		SOURCEFILE = sOURCEFILE;
	}
	
	public static int getHoldoutsetPrecentage() {
		return holdoutsetPrecentage;
	}

	public static void setHoldoutsetPrecentage(int holdoutsetPrecentage) {
		Globals.holdoutsetPrecentage = holdoutsetPrecentage;
	}
	
	public static double getNumberInstances() {
		return numberInstances;
	}

	public static void setNumberInstances(double numberInstances) {
		Globals.numberInstances = numberInstances;
	}
	
	public static int getNumClasses() {
		return numClasses;
	}

	public static void setNumClasses(int numClasses) {
		Globals.numClasses = numClasses;
	}

	public static int getNumAttributes() {
		return numAttributes;
	}

	public static void setNumAttributes(int numAttributes) {
		Globals.numAttributes = numAttributes;
	}
	
	public static boolean isDoBinaryClassification() {
		return doBinaryClassification;
	}

	public static void setDoBinaryClassification(boolean doBinaryClassification) {
		Globals.doBinaryClassification = doBinaryClassification;
	}

	public static String getObjectiveFunction() {
		return objectiveFunction;
	}

	public static void setObjectiveFunction(String objectiveFunction) {
		Globals.objectiveFunction = objectiveFunction;
	}

	public static String getDataStructureParameter() {
		return dataStructureParameter;
	}

	public static void setDataStructureParameter(String dataStructureParameter) {
		Globals.dataStructureParameter = dataStructureParameter;
	}
	
	public static int getHashParameter() {
		return hashParameter;
	}

	public static void setHashParameter(int hashParameter) {
		Globals.hashParameter = hashParameter;
	}
	
	public static int[] getParamsPerAtt() {
		return paramsPerAtt;
	}

	public static void setParamsPerAtt(int[] paramsPerAtt) {
		Globals.paramsPerAtt = paramsPerAtt;
	}

	public static boolean[] getIsNumericTrue() {
		return isNumericTrue;
	}

	public static void setIsNumericTrue(boolean[] isNumericTrue) {
		Globals.isNumericTrue = isNumericTrue;
	}
	
	public static boolean isDoDiscriminative() {
		return doDiscriminative;
	}

	public static void setDoDiscriminative(boolean doDiscriminative) {
		Globals.doDiscriminative = doDiscriminative;
	}
	
	public static Instances getCVInstances() {
		return CVInstances;
	}

	public static void setCVInstances(Instances cVInstances) {
		CVInstances = cVInstances;
	}
	
	public static Instances getStructure() {
		return structure;
	}

	public static void setStructure(Instances structure) {
		Globals.structure = structure;
	}
	
	public static String getAdaptiveControl() {
		return adaptiveControl;
	}

	public static void setAdaptiveControl(String adaptiveControl) {
		Globals.adaptiveControl = adaptiveControl;
	}

	public static double getAdaptiveControlParameter() {
		return adaptiveControlParameter;
	}

	public static void setAdaptiveControlParameter(double adaptiveControlParameter) {
		Globals.adaptiveControlParameter = adaptiveControlParameter;
	}

	
	public static void printWelcomeMessage() {
		String msg = "";
		msg += "------------------------------------------------------------------------- \n";
		msg += " Welcome to Aquila Audax \n";
		msg += " Version: " + getVersion() + "\n\n";
		msg += " Library for learning from extremely large quantities of data in minimal  \n";
		msg += " number of passes through the data. \n";
		msg += " Salient features: \n";
		msg += "         1) Superior Feature Engineering Capability \n";
		msg += "         2) Fast Optimization \n";
		msg += "         3) Out-of-core data processing \n";
		msg += "\n";
		msg += " Type -help for information how to use the library \n\n";
		msg += " Copyrights DataSmelly Pvt Ltd \n";
		msg += "------------------------------------------------------------------------- \n";
		
		System.out.println(msg);
	}
	
	public static void printWelcomeMessageWranglerini() {
		String msg = "";
		msg += "------------------------------------------------------------------------- \n";
		msg += " Invoking [Wranglerini] -- Data Wranglining Engine \n";
		msg += " Version: " + getVersion() + "\n\n";
		msg += " Copyrights DataSmelly Pvt Ltd \n";
		msg += "------------------------------------------------------------------------- \n";
		
		System.out.println(msg);
	}
	
	public static void printWelcomeMessageRecommendica() {
		String msg = "";
		msg += "------------------------------------------------------------------------- \n";
		msg += " Invoking [Recommendica] -- Recommender Systems Engine \n";
		msg += " Version: " + getVersion() + "\n\n";
		msg += " Copyrights DataSmelly Pvt Ltd \n";
		msg += "------------------------------------------------------------------------- \n";
		
		System.out.println(msg);
	}

	public static void printHelp() {
		String msg = "";
		msg += "------------------------------------------------------------------------- \n";
		msg += " java -jar AquilaAudax.jar  \n";
		msg += "         -typeofexperiment <traintest>  \n";
		msg += "               --trainfile  \n";
		msg += "               --testfile  \n";
		msg += "         -typeofexperiment <cv>  \n";
		msg += "               --trainfile  \n";
		msg += "         -typeofexperiment <prequential>  \n";
		msg += "         -typeofexperiment <recommender>  \n\n";
		msg += "         -numrounds  \n";
		msg += "         -numfolds  \n";
		msg += "------------------------------------------------------------------------- \n";
		
		System.out.println(msg);
	}

	public static boolean isNumInstancesKnown() {
		return numInstancesKnown;
	}

	public static void setNumInstancesKnown(boolean numInstancesKnown) {
		Globals.numInstancesKnown = numInstancesKnown;
	}

	public static int getPrequentialOutputResolution() {
		return prequentialOutputResolution;
	}

	public static void setPrequentialOutputResolution(int prequentialOutputResolution) {
		Globals.prequentialOutputResolution = prequentialOutputResolution;
	}
	
	public static boolean isDoMovingAverage() {
		return doMovingAverage;
	}

	public static void setDoMovingAverage(boolean doMovingAverage) {
		Globals.doMovingAverage = doMovingAverage;
	}

	public static int getPrequentialBufferOutputResolution() {
		return prequentialBufferOutputResolution;
	}

	public static void setPrequentialBufferOutputResolution(int prequentialBufferOutputResolution) {
		Globals.prequentialBufferOutputResolution = prequentialBufferOutputResolution;
	}

	public static boolean isComputeProbabilitiesFromCount() {
		return computeProbabilitiesFromCount;
	}

	public static void setComputeProbabilitiesFromCount(boolean computeProbabilitiesFromCount) {
		Globals.computeProbabilitiesFromCount = computeProbabilitiesFromCount;
	}

	public static String getTempDirectory() {
		return tempDirectory;
	}

	public static void setTempDirectory(String tempDirectory) {
		Globals.tempDirectory = tempDirectory;
	}

	public static String getOuputResultsDirectory() {
		return ouputResultsDirectory;
	}

	public static void setOuputResultsDirectory(String ouputResultsDirectory) {
		Globals.ouputResultsDirectory = ouputResultsDirectory;
	}

	public static String getFlowParameter() {
		return flowParameter;
	}

	public static void setFlowParameter(String flowParameter) {
		Globals.flowParameter = flowParameter;
	}

	public static boolean isGenerateDriftData() {
		return generateDriftData;
	}

	public static void setGenerateDriftData(boolean generateDriftData) {
		Globals.generateDriftData = generateDriftData;
	}
	
	public static int getTotalNInstancesBeforeDrift() {
		return totalNInstancesBeforeDrift;
	}

	public static void setTotalNInstancesBeforeDrift(int totalNInstancesBeforeDrift) {
		Globals.totalNInstancesBeforeDrift = totalNInstancesBeforeDrift;
	}

	public static int getTotalNInstancesDuringDrift() {
		return totalNInstancesDuringDrift;
	}

	public static void setTotalNInstancesDuringDrift(int totalNInstancesDuringDrift) {
		Globals.totalNInstancesDuringDrift = totalNInstancesDuringDrift;
	}

	public static int getTotalNInstancesAfterDrift() {
		return totalNInstancesAfterDrift;
	}

	public static void setTotalNInstancesAfterDrift(int totalNInstancesAfterDrift) {
		Globals.totalNInstancesAfterDrift = totalNInstancesAfterDrift;
	}

	public static int getDriftNAttributes() {
		return driftNAttributes;
	}

	public static void setDriftNAttributes(int driftNAttributes) {
		Globals.driftNAttributes = driftNAttributes;
	}

	public static int getDriftNAttributesValues() {
		return driftNAttributesValues;
	}

	public static void setDriftNAttributesValues(int driftNAttributesValues) {
		Globals.driftNAttributesValues = driftNAttributesValues;
	}

	public static double getDriftMagnitude() {
		return driftMagnitude;
	}

	public static void setDriftMagnitude(double driftMagnitude) {
		Globals.driftMagnitude = driftMagnitude;
	}

	public static String getFeatureSelection() {
		return featureSelection;
	}

	public static void setFeatureSelection(String featureSelection) {
		Globals.featureSelection = featureSelection;
	}

	public static double getFeatureSelectionParameter() {
		return featureSelectionParameter;
	}

	public static void setFeatureSelectionParameter(double featureSelectionParameter) {
		Globals.featureSelectionParameter = featureSelectionParameter;
	}

	public static boolean isDiscretizeOutOfCore() {
		return discretizeOutOfCore;
	}

	public static void setDiscretizeOutOfCore(boolean discretizeOutOfCore) {
		Globals.discretizeOutOfCore = discretizeOutOfCore;
	}

	public static String getDataSetName() {
		return dataSetName;
	}

	public static void setDataSetName(String dataSetName) {
		Globals.dataSetName = dataSetName;
	}

	public static String getPreProcessParameter() {
		return preProcessParameter;
	}

	public static void setPreProcessParameter(String preProcessParameter) {
		Globals.preProcessParameter = preProcessParameter;
	}

	public static String getIgnoreAttributes() {
		return ignoreAttributes;
	}

	public static void setIgnoreAttributes(String ignoreAttributes) {
		Globals.ignoreAttributes = ignoreAttributes;
	}

	public static int getClassAttribute() {
		return classAttribute;
	}

	public static void setClassAttribute(int classAttribute) {
		Globals.classAttribute = classAttribute;
	}

	public static String getAttributeType() {
		return attributeType;
	}

	public static void setAttributeType(String attributeType) {
		Globals.attributeType = attributeType;
	}

	public static int getDicedPercentage() {
		return dicedPercentage;
	}

	public static void setDicedPercentage(int dicedPercentage) {
		Globals.dicedPercentage = dicedPercentage;
	}

	public static boolean isDicedStratified() {
		return dicedStratified;
	}

	public static void setDicedStratified(boolean dicedStratified) {
		Globals.dicedStratified = dicedStratified;
	}

	public static int getDicedAt() {
		return dicedAt;
	}

	public static void setDicedAt(int dicedAt) {
		Globals.dicedAt = dicedAt;
	}

	public static Map<String, Double> getFsScore() {
		return fsScore;
	}

	public static void setFsScore(Map<String, Double> fsScore) {
		Globals.fsScore = fsScore;
	}

	public static String getCvFile() {
		return cvFile;
	}

	public static void setCvFile(String cvFile) {
		Globals.cvFile = cvFile;
	}

	public static File getCVFILE() {
		return CVFILE;
	}

	public static void setCVFILE(File cVFILE) {
		CVFILE = cVFILE;
	}

	public static boolean isCvFilePresent() {
		return cvFilePresent;
	}

	public static void setCvFilePresent(boolean cvFilePresent) {
		Globals.cvFilePresent = cvFilePresent;
	}

	public static String getDriftType() {
		return driftType;
	}

	public static void setDriftType(String driftType) {
		Globals.driftType = driftType;
	}

	public static int getNumRandAttributes() {
		return numRandAttributes;
	}

	public static void setNumRandAttributes(int numRandAttributes) {
		Globals.numRandAttributes = numRandAttributes;
	}

	public static int getLatentK() {
		return latentK;
	}

	public static void setLatentK(int latentK) {
		Globals.latentK = latentK;
	}

	public static double getDriftDelta() {
		return driftDelta;
	}

	public static void setDriftDelta(double driftDelta) {
		Globals.driftDelta = driftDelta;
	}

	public static int getDriftMagnitude2() {
		return driftMagnitude2;
	}

	public static void setDriftMagnitude2(int driftMagnitude2) {
		Globals.driftMagnitude2 = driftMagnitude2;
	}

	public static double getDriftMagnitude3() {
		return driftMagnitude3;
	}

	public static void setDriftMagnitude3(double driftMagnitude3) {
		Globals.driftMagnitude3 = driftMagnitude3;
	}

	public static boolean isPlotRMSEResuts() {
		return plotRMSEResuts;
	}

	public static void setPlotRMSEResuts(boolean plotRMSEResuts) {
		Globals.plotRMSEResuts = plotRMSEResuts;
	}

	public static String getDatasetRepository() {
		return datasetRepository;
	}

	public static void setDatasetRepository(String datasetRepository) {
		Globals.datasetRepository = datasetRepository;
	}



}
