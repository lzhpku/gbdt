package cn.edu.pku.gbdt;

import java.awt.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class Util {

	public static HashSet<Integer> sample(HashSet<Integer> set, int num) {
		HashSet<Integer> res = new HashSet<Integer>();
		ArrayList<Integer> t = new ArrayList<Integer>();
		for (Integer id : set) {
			t.add(id);
		}
		Random r = new Random();
		for (int i = 0; i < num; i ++) {
			int index = r.nextInt(t.size());
			res.add(t.get(index));
			t.remove(index);
		}
		return res;
	}
	
	public static HashSet<Integer> minusSet(HashSet<Integer> set, HashSet<Integer> subset) {
		HashSet<Integer> res = new HashSet<Integer>();
		for (Integer id : set) {
			if (!subset.contains(id)) {
				res.add(id);
			}
		}
		return res;
	}
	
	public static double computeMinLoss(ArrayList<Double> values) {
		if (values.size() < 2) {
			return 0;
		}
		double mean = 0;
		for (int i = 0; i < values.size(); i ++) {
			mean += values.get(i);
		}
		mean = mean / (double)values.size();
		double loss = 0;
		for (int i = 0; i < values.size(); i ++) {
			loss += (mean - values.get(i)) * (mean - values.get(i));
		}
		return loss;
	}
	
	public static Tree makeDecisionTree(DataSet dataset, HashSet<Integer> remainedSet,
			HashMap<Integer, Double> targets, int depth, HashSet<LeafNode> leafNodes,
			int maxDepth, double splitPoint) {
		if (depth < maxDepth) {
			ArrayList<String> attributes = dataset.getAttribute();
			double loss = -1.0;
			String selectedAttribute = null;
			double numConditionValue = 0;
			String strConditionValue = null;
			HashSet<Integer> selectedLeftIdSet = new HashSet<Integer>();
			HashSet<Integer> selectedRightIdSet = new HashSet<Integer>();
			for (int i = 0; i < attributes.size(); i ++) {
				boolean isRealType = dataset.isRealTypeField(attributes.get(i));
				if(isRealType) {
					HashSet<Double> numAttrValues = dataset.distinctValueset.get(attributes.get(i));
					if (splitPoint > 0 && numAttrValues.size() > splitPoint) {
						//TODO 实现随机抽样函数
					}
					for (Double attrValue : numAttrValues) {
						HashSet<Integer> leftIdSet = new HashSet<Integer>();
						HashSet<Integer> rightIdSet = new HashSet<Integer>();
						for (Integer id : remainedSet) {
							Instance instance = dataset.getInstance(id);
							double value = instance.numTypeFeature.get(attributes.get(i));
							if (value < attrValue) {
								leftIdSet.add(id);
							} else {
								rightIdSet.add(id);
							}
						}
						ArrayList<Double> leftTargets = new ArrayList<Double>();
						ArrayList<Double> rightTargets = new ArrayList<Double>();
						for (Integer id : leftIdSet) {
							leftTargets.add(targets.get(id));
						}
						for (Integer id : rightIdSet) {
							rightTargets.add(targets.get(id));
						}
						double sumLoss = computeMinLoss(leftTargets)
								+ computeMinLoss(rightTargets);
						if (loss < 0 || sumLoss < loss) {
							selectedAttribute = new String(attributes.get(i));
							numConditionValue = attrValue;
							loss = sumLoss;
							selectedLeftIdSet = (HashSet<Integer>) leftIdSet.clone();
							selectedRightIdSet = (HashSet<Integer>) rightIdSet.clone();
						}
					}
				} else {
					HashSet<String> strAttrValues = dataset.fieldType.get(attributes.get(i));
					for (String attrValue : strAttrValues) {
						HashSet<Integer> leftIdSet = new HashSet<Integer>();
						HashSet<Integer> rightIdSet = new HashSet<Integer>();
						for (Integer id : remainedSet) {
							Instance instance = dataset.getInstance(id);
							String value = instance.strTypeFeature.get(attributes.get(i));
							if (value.equals(attrValue)) {
								leftIdSet.add(id);
							} else {
								rightIdSet.add(id);
							}
						}
						ArrayList<Double> leftTargets = new ArrayList<Double>();
						ArrayList<Double> rightTargets = new ArrayList<Double>();
						for (Integer id : leftIdSet) {
							leftTargets.add(targets.get(id));
						}
						for (Integer id : rightIdSet) {
							rightTargets.add(targets.get(id));
						}
						double sumLoss = computeMinLoss(leftTargets)
								+ computeMinLoss(rightTargets);
						if (loss < 0 || sumLoss < loss) {
							selectedAttribute = new String(attributes.get(i));
							strConditionValue = new String(attrValue);
							loss = sumLoss;
							selectedLeftIdSet = (HashSet<Integer>) leftIdSet.clone();
							selectedRightIdSet = (HashSet<Integer>) rightIdSet.clone();
						}
					}
				}
			}
			if (selectedAttribute == null || loss < 0) {
				System.out.println("cannot determine the split attribute.");
			}
			Tree tree = new Tree();
			tree.splitFeature = new String(selectedAttribute);
			tree.isRealValueFeature = dataset.isRealTypeField(selectedAttribute);
			if (tree.isRealValueFeature) {
				tree.numConditionValue = numConditionValue;
			} else {
				tree.strConditionValue = new String(strConditionValue);
			}
			tree.leftTree = makeDecisionTree(dataset, selectedLeftIdSet, targets,
					depth + 1, leafNodes, maxDepth, 0);
			tree.rightTree = makeDecisionTree(dataset, selectedRightIdSet, targets,
					depth + 1, leafNodes, maxDepth, 0);
			return tree;
		} else {
			LeafNode node = new LeafNode(remainedSet);
			int K = dataset.getLabelSize();
			node.updatePredictValue(targets, K);
			leafNodes.add(new LeafNode(node));
			Tree tree = new Tree();
			tree.leafNode = new LeafNode(node);
			return tree;
		}
	}
}
