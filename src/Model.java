package cn.edu.pku.gbdt;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import cn.edu.pku.util.FileOutput;

public class Model {

	public int maxIter;
	public double sampleRate;
	public double learnRate;
	public int maxDepth;
	public int splitPoints;
	public HashMap<Integer, HashMap<String, Tree>> trees;
	
	public Model(int maxIter, double sampleRate, double learnRate, int maxDepth, int splitPoints) {
		super();
		this.maxIter = maxIter;
		this.sampleRate = sampleRate;
		this.learnRate = learnRate;
		this.maxDepth = maxDepth;
		this.splitPoints = splitPoints;
		this.trees = new HashMap<Integer, HashMap<String, Tree>>();
	}
	
	public void initialize(HashMap<Integer, HashMap<String, Double>> f, DataSet dataset) {
		for (Integer id : dataset.getInstanceIdset()) {
			HashMap<String, Double> t = new HashMap<String, Double>();
			for (String label : dataset.getLabelValueset()) {
				t.put(label, 0.0);
			}
			f.put(id, t);
		}
	}
	
	public HashMap<Integer, HashMap<String, Double>> computeResidual(
			DataSet dataset, HashSet<Integer> subset,
			HashMap<Integer, HashMap<String, Double>> f) {
		HashMap<Integer, HashMap<String, Double>> residual
			= new HashMap<Integer, HashMap<String, Double>>();
		HashSet<String> labelValueSet = dataset.getLabelValueset();
		for (Integer id : subset) {
			HashMap<String, Double> residualId = new HashMap<String, Double>();
			double pSum = 0.0;
			for (String label : labelValueSet) {
				pSum += Math.exp(f.get(id).get(label));
			}
			for (String label : labelValueSet) {
				double p = Math.exp(f.get(id).get(label)) / pSum;
				double y = 0.0;
				if (dataset.getInstance(id).strTypeFeature.get("label").equals(label)) {
					y = 1.0;
				}
				residualId.put(label, y - p);
			}
			residual.put(id, (HashMap<String, Double>) residualId.clone());
		}
		return residual;
	}
	
	public void updataFValue(HashMap<Integer, HashMap<String, Double>> f, Tree tree,
			HashSet<LeafNode> leafNodes, HashSet<Integer> subset, DataSet dataset,
			String label) {
		Set<Integer> dataIdset = dataset.getInstanceIdset();
		for (LeafNode node : leafNodes) {
			for (Integer id : node.getIdset()) {
				double t = f.get(id).get(label) + learnRate * node.getPredictValue();
				HashMap<String, Double> tt = f.get(id);
				tt.put(label, t);
				f.put(id, tt);
			}
		}
		for (Integer id : dataIdset) {
			if (!subset.contains(id)) {
				double t = f.get(id).get(label) + learnRate * tree.getPredictValue(dataset.getInstance(id));
				HashMap<String, Double> tt = f.get(id);
				tt.put(label, t);
				f.put(id, tt);
			}
		}
	}
	
	public HashMap<String, Double> computeInstanceFValue(Instance instance,
			HashSet<String> labelValueset) {
		HashMap<String, Double> fValue = new HashMap<String, Double> ();
		for (String label : labelValueset) {
			fValue.put(label, 0.0);
		}
		for (Integer iter : trees.keySet()) {
			for (String label : labelValueset) {
				Tree tree = trees.get(iter).get(label);
				double t = fValue.get(label) + learnRate * tree.getPredictValue(instance);
				fValue.put(label, t);
			}
		}
		return fValue;
	}
	
	public double computeLoss(DataSet dataset, HashSet<Integer> subset,
			HashMap<Integer, HashMap<String, Double>> f) {
		double loss = 0.0;
		for (Integer id : subset) {
			Instance instance = dataset.getInstance(id);
			HashMap<String, Double> fValues;
			if (f.containsKey(id)) {
				fValues = f.get(id);
			} else {
				fValues = computeInstanceFValue(instance, dataset.getLabelValueset());
			}
			HashMap<String, Double> expValues = new HashMap<String, Double>();
			double sumExpValues = 0.0;
			for (String label : fValues.keySet()) {
				double t = Math.exp(fValues.get(label));
				expValues.put(label, t);
				sumExpValues += t;
			}
			HashMap<String, Double> probs = new HashMap<String, Double>();
			for (String label : fValues.keySet()) {
				probs.put(label, Math.exp(fValues.get(label)) / sumExpValues);
			}
			loss -= Math.log(probs.get(instance.strTypeFeature.get("label")));
		}
		return loss / (double) subset.size();
	}
	
	public void train(DataSet dataset, HashSet<Integer> trainData,
			String outputPath, HashSet<Integer> testData) {
		FileOutput fo = new FileOutput(outputPath);
		HashSet<String> labelValueset = dataset.getLabelValueset();
		HashMap<Integer, HashMap<String, Double>> f = new HashMap<Integer, HashMap<String, Double>>();
		initialize(f, dataset);
		for (int iter = 0; iter < maxIter; iter ++) {
			HashSet<Integer> subset = (HashSet<Integer>) trainData.clone();
			if (sampleRate > 0 && sampleRate < 1) {
				subset = Util.sample(subset, (int)(subset.size() * sampleRate));
			}
			trees.put(iter, new HashMap<String, Tree>());
			HashMap<Integer, HashMap<String, Double>> residual
				= computeResidual(dataset, subset, f);
			for (String label : labelValueset) {
				HashSet<LeafNode> leafNodes = new HashSet<LeafNode>();
				HashMap<Integer, Double> targets = new HashMap<Integer, Double> ();
				for (Integer id : subset) {
					targets.put(id, residual.get(id).get(label));
				}
				Tree tree = Util.makeDecisionTree(dataset, subset, targets, 0, leafNodes, maxDepth, splitPoints);
				HashMap<String, Tree> t = trees.get(iter);
				t.put(label, tree);
				trees.put(iter, t);
				updataFValue(f, tree, leafNodes, subset, dataset, label);
			}
			double accuracy = 0.0, aveRisk = 0.0;
			if (testData != null && testData.size() != 0.0) {
				String res = test(dataset, testData);
				accuracy = Double.parseDouble(res.substring(0, res.indexOf(" ")));
				aveRisk = Double.parseDouble(res.substring(res.indexOf(" ") + 1));
			}
			double trainLoss = computeLoss(dataset, trainData, f);
			double testLoss = computeLoss(dataset, testData, f);
			System.out.println("accuracy=" + accuracy
						+ "  average train loss=" + trainLoss
						+ "  average test loss=" + testLoss
						+ "  average risk=" + aveRisk);
			System.out.println((iter + 1) + "th iteration completed");
			System.out.println();
			try {
				fo.t3.write("accuracy=" + accuracy
						+ "  average train loss=" + trainLoss
						+ "  average test loss=" + testLoss
						+ "  average risk=" + aveRisk);
				fo.t3.newLine();
				fo.t3.write((iter + 1) + "th iteration completed");
				fo.t3.newLine();
				fo.t3.newLine();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		fo.closeOutput();
	}
	
	public String test(DataSet dataset, HashSet<Integer> testData) {
		int rightPrediction = 0;
		HashSet<String> labelValueset = dataset.getLabelValueset();
		double risk = 0.0;
		for (Integer id : testData) {
			Instance instance = dataset.getInstance(id);
			HashMap<String, Double> probs = predict(instance, labelValueset);
			String predictLabel = predictLabel(probs);
			double singleRisk = 0.0;
			for (String label : probs.keySet()) {
				if (label.equals(instance.strTypeFeature.get("label"))) {
					singleRisk += (1.0 - probs.get(label));
				} else {
					singleRisk += probs.get(label);
				}
			}
			risk += singleRisk / (double) probs.size();
			if (instance.strTypeFeature.get("label").equals(predictLabel)) {
				rightPrediction ++;
			}
		}
		return String.valueOf(((double) rightPrediction / (double) testData.size())
				+ " " + String.valueOf(risk / (double) testData.size()));
	}
	
	public HashMap<String, Double> predict(Instance instance, HashSet<String> labelValueset) {
		HashMap<String, Double> fValue = computeInstanceFValue(instance, labelValueset);
		HashMap<String, Double> expValues = new HashMap<String, Double> ();
		double expSum = 0.0;
		for (String label : fValue.keySet()) {
			double t = Math.exp(fValue.get(label));
			expValues.put(label, t);
			expSum += t;
		}
		HashMap<String, Double> probs = new HashMap<String, Double>();
		for (String label : expValues.keySet()) {
			probs.put(label, expValues.get(label) / expSum);
		}
		return probs;
	}
	
	public String predictLabel(HashMap<String, Double> probs) {
		String predictLabel = null;
		for (String label : probs.keySet()) {
			if (predictLabel == null || probs.get(label) > probs.get(predictLabel)) {
				predictLabel = label;
			}
		}
		return predictLabel;
	}
	
}
