package utils;

public class SanctityCheck {

	public static void printExperimentInformation() {

		String val = Globals.getExperimentType();

		String msg = "";

		msg += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n";
		msg += "Experiment type: " + val + "\n";
		msg += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n";

		if (val.equalsIgnoreCase("cv")) {

			msg += getCVExperimentinfo();

		} else if (val.equalsIgnoreCase("traintest")) {

		} else if (val.equalsIgnoreCase("recommender")) {

		} else if (val.equalsIgnoreCase("prequential")) {

			msg += getPrequentialExperimentinfo();

		} else if (val.equalsIgnoreCase("flowMachines")) {

		} else if (val.equalsIgnoreCase("drift")) {

		} else if (val.equalsIgnoreCase("preprocess")) {

		} else if (val.equalsIgnoreCase("external")) {

		} else if (val.equalsIgnoreCase("moa")) {

		}

		msg += "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ \n";
		msg += "That is all what AquilaAudax needs to train this model -- Ignoring all other flags \n";
		msg += "Sanctity Check of input arguments is now complete. \n\n";
		System.out.println(msg);
	}

	private static String getPrequentialExperimentinfo() {
		String msg = "";
		msg += "Num Exp = " + Globals.getNumExp() + "\n";

		String val = Globals.getModel();

		if (val.equalsIgnoreCase("AnDE")) {

			msg += getAnDEModeinfo();

		} else if (val.equalsIgnoreCase("ALR")) {

			msg += getALRModeinfo();

		} else if (val.equalsIgnoreCase("hALR")) {
			
			msg += getHALRModeinfo();

		} else if (val.equalsIgnoreCase("KDB")) {

		} else if (val.equalsIgnoreCase("FM")) {

		} else if (val.equalsIgnoreCase("ANN")) {

		}

		msg += "Adaptive Control = " + Globals.getAdaptiveControl() + " \n";
		msg += "Adaptive Control parameter = " + Globals.getAdaptiveControlParameter()  + " \n";

		return msg;
	}

	private static String getCVExperimentinfo() {
		String msg = "";
		msg += "Num Exp = " + Globals.getNumExp() + ", Num of Folds = " + Globals.getNumFolds() + "\n";

		String val = Globals.getModel();

		if (val.equalsIgnoreCase("AnDE")) {

			msg += getAnDEModeinfo();

		} else if (val.equalsIgnoreCase("ALR")) {

			msg += getALRModeinfo();

		} else if (val.equalsIgnoreCase("hALR")) {
			
			msg += getHALRModeinfo();

		} else if (val.equalsIgnoreCase("feALR")) {
			
			msg += getFEALRModeinfo();

		} else if (val.equalsIgnoreCase("KDB")) {

		} else if (val.equalsIgnoreCase("FM")) {

		} else if (val.equalsIgnoreCase("ANN")) {

		}

		return msg;
	}

	private static String getALRModeinfo() {
		String msg = "";
		msg += "ALR -- Level (n) = " + Globals.getLevel() + "\n";
		msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n";

		String val = Globals.getOptimizer();

		if (val.equalsIgnoreCase("None")) {

			msg += "Optimizer = None: Training a Generative Classifier (also known as AnJE) \n";

		} else if (val.equalsIgnoreCase("SGD")) {

			msg += "Do WANBIA-C trick = " + Globals.isDoWanbiac() + "\n";

			msg += getSGDOptimizereinfo();

		} else if (val.equalsIgnoreCase("MCMC")) {

			msg += getMCMCOptimizereinfo();

		} 

		return msg;
	}
	
	private static String getHALRModeinfo() {
		String msg = "";
		msg += "hALR -- Level (n) = " + Globals.getLevel() + "\n";
		msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n";

		String val = Globals.getOptimizer();

		if (val.equalsIgnoreCase("None")) {

			msg += "Optimizer = None: Training a Generative Classifier (also known as AnJE) \n";

		} else if (val.equalsIgnoreCase("SGD")) {

			msg += "Do WANBIA-C trick = " + Globals.isDoWanbiac() + "\n";

			msg += getSGDOptimizereinfo();

		} else if (val.equalsIgnoreCase("MCMC")) {

			msg += getMCMCOptimizereinfo();

		} 

		return msg;
	}
	
	private static String getFEALRModeinfo() {
		String msg = "";
		msg += "feALR -- Level to (n) = " + Globals.getLevel() + "\n";
		msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n";

		String val = Globals.getOptimizer();

		if (val.equalsIgnoreCase("None")) {

			msg += "Optimizer = None: Training a Generative Classifier (also known as AnJE) \n";

		} else if (val.equalsIgnoreCase("SGD")) {

			msg += "Do WANBIA-C trick = " + Globals.isDoWanbiac() + "\n";

			msg += getSGDOptimizereinfo();

		} else if (val.equalsIgnoreCase("MCMC")) {

			msg += getMCMCOptimizereinfo();

		} 

		return msg;
	}

	private static String getAnDEModeinfo() {
		String msg = "";
		msg += "AnDE -- Level (n) = " + Globals.getLevel() + "\n";
		msg += "Parameter Structure = " + Globals.getDataStructureParameter() + "\n";

		return msg;
	}

	private static String getMCMCOptimizereinfo() {
		String msg = "";
		msg += "MCMC optimizer type = " + Globals.getMcmcType() + "\n";
		return msg;
	}

	private static String getSGDOptimizereinfo() {
		String msg = "";
		if (Globals.isDoDiscriminative()) {
			msg += "SGD type = " + Globals.getSgdType() + "\n";
			msg += "Regularization = " + Globals.isDoRegularization() + "\n";

			if (Globals.isDoRegularization()) { 
				msg += "Regularization Type = " +  0 + "\n";
				msg += "Regularization Towards = " +  0 + "\n";
				msg += "Lambda = " + Globals.getLambda() + "\n";
				msg += "Adaptive Regularization = " + Globals.getAdaptiveRegularization() + "\n";
			}

			msg += "Tuning Parameter = " + Globals.getSgdTuningParameter() + "\n";
			msg += "Cross-validate Tuning Parameter = " + Globals.isDoCrossValidateTuningParameter() + "\n";
		}
		return msg;
	}

	public static boolean checkStringArgs() {

		if (checkExperimentType() && checkModel() && checkOptimizer() 
				&& checkSGDType() && checkAdaptiveRegularization()
				&& checkMCMCType() && checkRegularizationType() 
				&& checkDiscretizationType()
				&& checkObjectiveFunctionType()
				&& checkStructureParameter()
				&& checkAdaptiveControl()
				&& checkFeatureSelection()
				&& checkPreProcessParameterType())
			return true;
		else 
			return false;


	}
	
	private static boolean checkFeatureSelection() {

		boolean flag = false;
		String val = Globals.getFeatureSelection();

		if (val.equalsIgnoreCase("None") || val.equalsIgnoreCase("Count") || val.equalsIgnoreCase("MI") || 
				val.equalsIgnoreCase("ChiSqTest") || val.equalsIgnoreCase("GTest") ||
				val.equalsIgnoreCase("FisherExactTest") || val.equalsIgnoreCase("AnJETest") ||
				val.equalsIgnoreCase("AnJETestLOOCV") || val.equalsIgnoreCase("ALRTest")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-featureSelection takes values from {None, Count, MI, ChiSqTest, GTest, FisherExactTest, AnJETest, AnJETestLOOCV, ALRTest}");
		}

		return flag;
	}
	
	private static boolean checkAdaptiveControl() {

		boolean flag = false;
		String val = Globals.getAdaptiveControl();

		if (val.equalsIgnoreCase("None") || val.equalsIgnoreCase("Decay") || 
				val.equalsIgnoreCase("Window")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-adaptivecontrol takes values from {None, Decay, Window}");
		}

		return flag;
	}

	private static boolean checkObjectiveFunctionType() {

		boolean flag = false;
		String val = Globals.getObjectiveFunction();

		if (val.equalsIgnoreCase("CLL") || val.equalsIgnoreCase("MSE") || 
				val.equalsIgnoreCase("HL")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-objectiveFunction takes values from {CLL, MSE, HLL}");
		}

		return flag;
	}

	private static boolean checkStructureParameter() {

		boolean flag = false;
		String val = Globals.getDataStructureParameter();

		if (val.equalsIgnoreCase("Flat") || val.equalsIgnoreCase("IndexedBig") || 
				val.equalsIgnoreCase("BitMap") || val.equalsIgnoreCase("Hash")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-structureParameter takes values from {Flat, IndexedBig, BitMap, Hash}");
		}

		return flag;
	}

	private static boolean checkDiscretizationType() {

		boolean flag = false;
		String val = Globals.getDiscretization();

		if (val.equalsIgnoreCase("None")) {
			flag = true;
		} else if (val.equalsIgnoreCase("mdl")) {
			flag = true;
		} else if (val.equalsIgnoreCase("ef")) {
			flag = true;
		} else if (val.equalsIgnoreCase("ew")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-DiscretizationType type takes values from {None, mdl, ef, ew}");
		}

		if (!val.equalsIgnoreCase("None") && Globals.isNormalizeNumeric()) {
			System.out.println("Can't discretize and normalize at the same time");
			flag = false;
		}

		return flag;
	}

	private static boolean checkMCMCType() {

		boolean flag = false;
		String val = Globals.getMcmcType();

		if (val.equalsIgnoreCase("ALS")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-MCMC type takes values from {ALS}");
		}

		return flag;
	}

	private static boolean checkExperimentType() {

		boolean flag = false;
		String val = Globals.getExperimentType();

		if (val.equalsIgnoreCase("drift")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-experimentType takes values from {drift}");
		}

		return flag;
	}

	private static boolean checkModel() {

		boolean flag = false;
		String val = Globals.getModel();

		if (val.equalsIgnoreCase("AnDE") || 
				val.equalsIgnoreCase("ALR") || val.equalsIgnoreCase("KDB") ||
				val.equalsIgnoreCase("FM") || val.equalsIgnoreCase("ANN") ||
				val.equalsIgnoreCase("hALR") || val.equalsIgnoreCase("feALR") || val.equalsIgnoreCase("feeALR")) {
			flag = true;
		}  else {
			flag = false;
			System.out.println("-Model takes values from {AnDE, ALR, KDB, FM, ANN, hALR, feALR, feeALR}");
		}

		return flag;
	}

	private static boolean checkOptimizer() {

		boolean flag = false;
		String val = Globals.getOptimizer();

		if (val.equalsIgnoreCase("None") || val.equalsIgnoreCase("SGD") || val.equalsIgnoreCase("MCMC")) {
			flag = true;
		}  else {
			flag = false;
			System.out.println("-optimizer takes values from {None, SGD, MCMC}");
		}

		return flag;
	}

	private static boolean checkSGDType() {

		boolean flag = false;
		String val = Globals.getSgdType();

		if (val.equalsIgnoreCase("plainsgd") || val.equalsIgnoreCase("adagrad") ||
				val.equalsIgnoreCase("adadelta") || val.equalsIgnoreCase("nplr")) {
			flag = true;
		}  else {
			flag = false;
			System.out.println("-sgdType takes values from {plainsgd, adagrad, adadelta, nplr}");
		}

		return flag;
	}

	private static boolean checkAdaptiveRegularization() {

		boolean flag = false;
		String val = Globals.getAdaptiveRegularization();

		if (val.equalsIgnoreCase("none") || val.equalsIgnoreCase("rendal") || val.equalsIgnoreCase("provost") ||
				val.equalsIgnoreCase("zaidi1") || val.equalsIgnoreCase("zaidi2")) {
			flag = true;
		}  else {
			flag = false;
			System.out.println("-adaptiveRegularization takes values from {none, rendal, provost, zaidi1, zaidi2}");
		}

		return flag;
	}

	private static boolean checkRegularizationType() {

		boolean flag = false;
		String val = Globals.getRegularizationType();

		if (val.equalsIgnoreCase("L2")) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-Regularization Type takes values from {L2}");
		}

		return flag;
	}
	
	private static boolean checkPreProcessParameterType() {

		boolean flag = false;
		String val = Globals.getPreProcessParameter();
		
		if (val.equalsIgnoreCase("Explore") || val.equalsIgnoreCase("CreateHeader") || val.equalsIgnoreCase("Discretize") 
				|| val.equalsIgnoreCase("Slice") || val.equalsIgnoreCase("Normalize") || val.equalsIgnoreCase("MissingImputate")
				|| val.equalsIgnoreCase("HeaderAlignment") || val.equalsIgnoreCase("Dice") || val.equalsIgnoreCase("Binarize")
				|| val.equalsIgnoreCase("BinarizeClass") || val.equalsIgnoreCase("compactDiscretizedFile") ) {
			flag = true;
		} else {
			flag = false;
			System.out.println("-preProcessParameter takes values from <Explore, CreateHeader, Discretize, Slice, Normalize, MissingImputate, HeaderAlignment, Dice, Binarize, BinarizeClass, compactDiscretizedFile}");
		}

		return flag;
	}

	public static String determineFlowVal() {

		String flowVal = "";

		String val = Globals.getFlowParameter();

		String[] parseVal = val.split("\\s*:\\s*");

		if (parseVal.length != 2) {
			flowVal = "";
		}  else {
			flowVal = parseVal[0];
		}

		return flowVal;
	}

	public static double[] getFlowValues() {

		String flowValues = "";

		String val = Globals.getFlowParameter();

		String[] parseVal = val.split("\\s*:\\s*");

		if (parseVal.length != 2) {
			flowValues = "";
		}  else {
			flowValues = parseVal[1];
		}

		flowValues = flowValues.replaceAll("[{()}]", "");

		String[] parseFlowValues = flowValues.split("\\s*,\\s*");

		int numValues = parseFlowValues.length;

		double[] flowValuesDouble = null;

		if (numValues != 0) {
			flowValuesDouble = new double[numValues];

			for (int i = 0; i < parseFlowValues.length; i++) {
				flowValuesDouble[i] = Double.parseDouble(parseFlowValues[i]);
			}
		}
		
		return flowValuesDouble;
	}

	public static String determineDSValue() {
		String val = Globals.getTrainFile();
		int loclast = val.lastIndexOf("/");
		int locDot = val.lastIndexOf(".");
		String ds = val.substring(loclast+1, locDot);
		return ds;
	}

}
