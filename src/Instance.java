package cn.edu.pku.gbdt;

import java.util.HashMap;

public class Instance {
	
	public HashMap<String, Double> numTypeFeature;
	public HashMap<String, String> strTypeFeature;
	
	
	
	public Instance() {
		super();
		numTypeFeature = new HashMap<String, Double>();
		strTypeFeature = new HashMap<String, String>();
	}

	public Instance(HashMap<String, Double> numTypeFeature, HashMap<String, String> strTypeFeature) {
		super();
		this.numTypeFeature = numTypeFeature;
		this.strTypeFeature = strTypeFeature;
	}

}
