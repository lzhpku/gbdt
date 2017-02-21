package cn.edu.pku.gbdt;

public class Tree {

	public LeafNode leafNode;
	public Tree leftTree;
	public Tree rightTree;
	public String splitFeature;
	public double numConditionValue;
	public String strConditionValue;
	public boolean isRealValueFeature;
	
	public Tree() {
		super();
		leafNode = null;
		leftTree = null;
		rightTree = null;
		splitFeature = null;
		numConditionValue = 0;
		strConditionValue = null;
		isRealValueFeature = true;
	}
	
	public double getPredictValue(Instance instance) {
		if (leafNode != null) {
			return leafNode.getPredictValue();
		}
		if (splitFeature == null) {
			System.out.println("the tree is null");
		}
		if (isRealValueFeature && instance.numTypeFeature.get(splitFeature) < numConditionValue) {
			return leftTree.getPredictValue(instance);
		} else if (!isRealValueFeature && instance.strTypeFeature.get(splitFeature).equals(strConditionValue)) {
			return leftTree.getPredictValue(instance);
		}
		return rightTree.getPredictValue(instance);
	}
	
	public String describe() {
		if (leftTree == null || rightTree == null) {
			return leafNode.describe();
		}
		String leftInfo = leftTree.describe();
		String rightInfo = rightTree.describe();
		return "{split feature:" + splitFeature
				+ ", split value:" + numConditionValue + "," + strConditionValue
				+ "[left tree:" + leftInfo + "," + "right tree:" + rightInfo + "]}";
	}
}