package cn.edu.pku.gbdt;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class LeafNode {
	
	public HashSet<Integer> idset;
	public double predictValue;
	
	public LeafNode(LeafNode node) {
		this.idset = (HashSet<Integer>) node.idset.clone();
		this.predictValue = node.predictValue;
	}
	
	public LeafNode(HashSet<Integer> idset) {
		this.idset = (HashSet<Integer>) idset.clone();
		predictValue = 0;
	}

	public HashSet<Integer> getIdset() {
		return idset;
	}

	public void setIdset(HashSet<Integer> idset) {
		this.idset = idset;
	}

	public double getPredictValue() {
		return predictValue;
	}

	public void setPredictValue(double predictValue) {
		this.predictValue = predictValue;
	}
	
	public void updatePredictValue(HashMap<Integer, Double> targets, int K) {
		double sum1 = 0;
		for (Integer id : idset) {
			sum1 += targets.get(id);
		}
		if (sum1 == 0) {
			predictValue = 0.0;
			return;
		}
		double sum2 = 0;
		for (Integer id : idset) {
			double num = Math.abs(targets.get(id));
			sum2 += num * (1.0 - num);
		}
		try {
			predictValue = (double)(K - 1) / (double) K * (sum1 / sum2);
		} catch (Exception e) {
			System.out.println("zero division");
		}
	}
	
	public String describe() {
		return "{LeafNode:" + predictValue + "}";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((idset == null) ? 0 : idset.hashCode());
		long temp;
		temp = Double.doubleToLongBits(predictValue);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		LeafNode other = (LeafNode) obj;
		if (idset == null) {
			if (other.idset != null)
				return false;
		} else if (!idset.equals(other.idset))
			return false;
		if (Double.doubleToLongBits(predictValue) != Double.doubleToLongBits(other.predictValue))
			return false;
		return true;
	}
	
	
}
