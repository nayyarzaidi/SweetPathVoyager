package utils;

import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.io.File;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYErrorRenderer;
import org.jfree.data.xy.XYIntervalSeries;
import org.jfree.data.xy.XYIntervalSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import evaluation.evaluationDrift;

public class Plotting {

	public static void plot(String fileName, int numFlows, double[] level, double driftMagnitude, 
			double[] decayWindow_Rate, String dw_name, int nSamplesForChart,
			double[][][] meanErrorPlot, double[][][] stdDevErrorPlot, double[] minMean) { 

		// now ready to plot
		for (int c = 0; c < numFlows; c++) {

			String learnername = "";
			learnername = Globals.getModel() + " - " + level[c]+ " - Drift (" + Globals.getDriftMagnitude() + ", " + Globals.getDriftMagnitude2()  + "," + Globals.getDriftMagnitude3()+ ")";

			//PrintWriter csv = new PrintWriter(new BufferedWriter(new FileWriter(new File(fileName + ".csv"))));
			//csv.println("magnitude,nInstances,mean error,stddev-error");

			XYSeriesCollection dataSet = new XYSeriesCollection();
			XYIntervalSeriesCollection dataSetErrors = new XYIntervalSeriesCollection();
			for (int m = 0; m < decayWindow_Rate.length; m++) {
				// one series per magnitude

				//XYSeries series = new XYSeries("Magnitude=" + mags[m]);
				//XYIntervalSeries seriesError = new XYIntervalSeries("Magnitude=" + mags[m]);

				XYSeries series = new XYSeries(dw_name + decayWindow_Rate[m]);
				XYIntervalSeries seriesError = new XYIntervalSeries(dw_name + decayWindow_Rate[m]);

				for (int indexPlot = 0; indexPlot < nSamplesForChart; indexPlot++) {
					int trueNInstancesForIndex = 1 + indexPlot * Globals.getPrequentialBufferOutputResolution();
					double mean = meanErrorPlot[c][m][indexPlot];

					double low = mean - stdDevErrorPlot[c][m][indexPlot];
					double high = mean + stdDevErrorPlot[c][m][indexPlot];
					seriesError.add(trueNInstancesForIndex, trueNInstancesForIndex, trueNInstancesForIndex, mean, low, high);
					series.add(trueNInstancesForIndex, mean);
					//csv.println(decayWindow_Rate[m] + "," + trueNInstancesForIndex + "," + meanErrorPlot[c][m][indexPlot] + ","+ stdDevErrorPlot[c][m][indexPlot]);
				}
				dataSet.addSeries(series);
				dataSetErrors.addSeries(seriesError);
			}
			//csv.close();

			//JFreeChart chart = ChartFactory.createXYBarChart(null, learners.get(c).getClass().getSimpleName(), false, "error-rate", dataSetErrors, PlotOrientation.VERTICAL, true, false, false);
			JFreeChart chart = ChartFactory.createXYBarChart(null, learnername, false, "error-rate", dataSetErrors, PlotOrientation.VERTICAL, true, false, false);

			XYPlot plot = chart.getXYPlot();
			NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
			// yAxis.setLowerBound(minOverallStd);

			XYErrorRenderer errorRenderer = new XYErrorRenderer();
			errorRenderer.setDrawXError(false);
			errorRenderer.setLinesVisible(true);
			Ellipse2D.Double ellipse = new Ellipse2D.Double(-1, -1, 2, 2);
			for (int i = 0; i < dataSetErrors.getSeriesCount(); i++) {
				errorRenderer.setSeriesShape(i, ellipse);
			}
			plot.setRenderer(errorRenderer);
			plot.setBackgroundPaint(new Color(235, 235, 235));

			// remove grid
			plot.setDomainGridlinesVisible(false);
			plot.setRangeGridlinesVisible(false);

			//ChartTools.saveChartAsPDF(chart, new File(fileName + "-error.pdf"));

			//chart = ChartFactory.createXYLineChart(Double.toString(flowValues[c]), "nInstances", "error-rate", dataSet, PlotOrientation.VERTICAL, true, false, false);
			chart = ChartFactory.createXYLineChart(learnername, "nInstances", "error-rate", dataSet, PlotOrientation.VERTICAL, true, false, false);

			plot = chart.getXYPlot();
			yAxis = (NumberAxis) plot.getRangeAxis();
			double min = Math.max(0.0, minMean[c] - .01);

			//yAxis.setLowerBound(min);
			// yAxis.setUpperBound(1.0);

			yAxis.setUpperBound(1.0);
			yAxis.setLowerBound(0.0);

			//yAxis.setUpperBound(0.1);
			//yAxis.setLowerBound(-0.01);

			// remove grid
			plot.setDomainGridlinesVisible(false);
			plot.setRangeGridlinesVisible(false);

			String filename = fileName + "_Level_" + level[c] + ".plot";
			
			ChartTools.saveChartAsPDF(chart, new File(filename + "-mean.pdf"));
		} 

	}

	public static void plot2(String fileName, int numFlows, double[] level, double driftMagnitude, 
			double[] decayWindow_Rate, String dw_name, int nSamplesForChart,
			double[][][] meanErrorPlot, double[][][] stdDevErrorPlot, double[] minMean) {

		String learnername = "";
		learnername = "Drift (" + Globals.getDriftMagnitude() + ", " + Globals.getDriftMagnitude2()  + "," + Globals.getDriftMagnitude3()+ ")";
		
		fileName += ".plot";

		XYSeriesCollection dataSet = new XYSeriesCollection();

		for (int c = 0; c < numFlows; c++) {

			for (int m = 0; m < decayWindow_Rate.length; m++) {
				XYSeries series = new XYSeries(dw_name + decayWindow_Rate[m] + ", Level=" + level[c]);

				for (int indexPlot = 0; indexPlot < nSamplesForChart; indexPlot++) {
					int trueNInstancesForIndex = 1 + indexPlot * Globals.getPrequentialBufferOutputResolution();
					double mean = meanErrorPlot[c][m][indexPlot];
					series.add(trueNInstancesForIndex, mean);
				}

				dataSet.addSeries(series);
			}

			JFreeChart chart = ChartFactory.createXYLineChart(learnername, "nInstances", "error-rate", dataSet, PlotOrientation.VERTICAL, true, false, false);

			XYPlot plot = chart.getXYPlot();

			NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();

			//			LogAxis logAxis = new LogAxis();
			//			logAxis.setMinorTickMarksVisible(true);
			//			logAxis.setAutoRange(true);
			//			logAxis.setRange(0, 1);

			plot.setRangeAxis(yAxis);

			//double min = Math.max(0.0, minMean[c] - .01);
			//yAxis.setLowerBound(min);
			// yAxis.setUpperBound(1.0);

			//yAxis.setUpperBound(1.0);
			//yAxis.setLowerBound(0.0);

			// remove grid
			plot.setDomainGridlinesVisible(false);
			plot.setRangeGridlinesVisible(false);
			
			ChartTools.saveChartAsPDF(chart, new File(fileName + "-mean.pdf"));
		} 

	}

}
