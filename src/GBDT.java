package cn.edu.pku.gbdt;

import java.util.HashSet;

public class GBDT {

	public static void main(String[] args) {
		DataSet dataset = new DataSet("data/adult.data.csv");
		dataset.describe();
		String outputPath = "stat.txt";
		Model model = new Model(10, 0.3, 0.1, 1, 0);
		HashSet<Integer> trainData = Util.sample(dataset.getInstanceIdset(), dataset.instances.size() * 2 / 3);
		HashSet<Integer> testData = Util.minusSet(dataset.getInstanceIdset(), trainData);
		model.train(dataset, trainData, outputPath, testData);
	}
}
