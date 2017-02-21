package cn.edu.pku.gbdt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import cn.edu.pku.util.FileInput;

public class DataSet {
	
	public HashMap<Integer, Instance> instances;
	public HashMap<String, HashSet<Double>> distinctValueset;
	public ArrayList<String> fieldNames;
	public HashMap<String, HashSet<String>> fieldType;
	
	public DataSet (String fileName) {
		int lineCnt = 0;
		instances = new HashMap<Integer, Instance>();
		distinctValueset = new HashMap<String, HashSet<Double>>();
		FileInput fi = new FileInput(fileName);
		String line = new String ();
		try {
			while((line = fi.reader.readLine()) != null) {
				if (line.equals("\n")) {
					continue;
				}
				//更换数据之后需要更改****************
//				line = line.substring(0, line.length() - 1);
				String[] fields = line.split(",");
				//更换数据之后需要更改****************
				//表头
				if (lineCnt == 0) {
					fieldNames = new ArrayList<String>();
					for (int i = 0; i < fields.length; i ++) {
						fieldNames.add(fields[i]);
					}
				} else {
					if (fields.length != fieldNames.size()) {
						System.out.println("wrong field number: line " + lineCnt);
						continue;
					}
					if (lineCnt == 1) {
						fieldType = new HashMap<String, HashSet<String>>();
						for (int i = 0; i < fieldNames.size(); i ++) {
							HashSet<String> valueSet = new HashSet<String>();
							try {
								Double.parseDouble(fields[i]);
								distinctValueset.put(
										fieldNames.get(i), new HashSet<Double>());
							} catch (Exception e) {
								valueSet.add(fields[i]);
							}
							fieldType.put(fieldNames.get(i), (HashSet<String>) valueSet.clone());
						}
					}
					instances.put(lineCnt, makeInstance(fields));
				}
				lineCnt ++;
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		fi.closeInput();
	}
	
	public Instance makeInstance(String[] fields) {
		Instance instance = new Instance();
		for (int i = 0; i < fields.length; i ++) {
			String fieldName = fieldNames.get(i);
			if (isRealTypeField(fieldName)) {
				try {
					double value = Double.parseDouble(fields[i]);
					instance.numTypeFeature.put(fieldName, value);
					HashSet<Double> t = distinctValueset.get(fieldName);
					t.add(value);
					distinctValueset.put(fieldName, t);
				} catch (Exception e) {
					System.out.println("value type conflict");
				}
			} else {
				instance.strTypeFeature.put(fieldName, fields[i]);
				HashSet<String> t = fieldType.get(fieldName);
				t.add(fields[i]);
				fieldType.put(fieldName, t);
			}
		}
		return instance;
	}
	
	public void describe() {
		String info = "feature:";
		for (int i = 0; i < fieldNames.size(); i ++) {
			info += " " + fieldNames.get(i);
		}
		info += "\ndatasize: " + instances.size() + "\n";
		for (int i = 0; i < fieldNames.size(); i ++) {
			String fieldName = fieldNames.get(i);
			info += "\ndescription for field:" + fieldName + "\n";
			if (isRealTypeField(fieldNames.get(i))) {
				HashSet<Double> t = distinctValueset.get(fieldName);
				info += "real value, distinct values number:" + t.size();
				double maxValue = -0x7fffffff, minValue = 0x7777777;
				for (Double num : t) {
					if (maxValue < num) {
						maxValue = num;
					}
					if (minValue > num) {
						minValue = num;
					}
				}
				info += "\nrange [" + String.valueOf(minValue) + ", "
						+ String.valueOf(maxValue) + "]\n";
			} else {
				HashSet<String> t = fieldType.get(fieldName);
				info += "enum type, distinct values number:" + t.size();
				info += "\nvalue [";
				for (String value : t) {
					info += value + ", ";
				}
				info += "]\n";
			}
		}
		System.out.println(info);
	}
	
	public boolean isRealTypeField(String name) {
		if (!fieldNames.contains(name)) {
			System.out.println("field name not in the dictionary");
		}
		return fieldType.get(name).size() == 0;
	}
	
	public HashSet<Integer> getInstanceIdset() {
		HashSet<Integer> res = new HashSet<Integer>();
		for (Integer id : instances.keySet()) {
			res.add(id);
		}
		return res;
	}
	
	public int getLabelSize() {
		return fieldType.get("label").size();
	}
	
	public HashSet<String> getLabelValueset() {
		return fieldType.get("label");
	}
	
	public Instance getInstance(int id) {
		return instances.get(id);
	}
	
	public ArrayList<String> getAttribute() {
		ArrayList<String> ret = new ArrayList<String>();
		for (int i = 0; i < fieldNames.size(); i ++) {
			if (!fieldNames.get(i).equals("label")) {
				ret.add(fieldNames.get(i));
			}
		}
		return ret;
	}
	
	public static void main(String [] args) {
		DataSet dataset = new DataSet("../GBDT/data/adult.data.csv");
		dataset.describe();
	}
}

