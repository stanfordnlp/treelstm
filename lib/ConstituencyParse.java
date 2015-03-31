import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Trees;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;

public class ConstituencyParse {

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    if (!props.containsKey("tokpath") ||
        !props.containsKey("parentpath")) {
      System.err.println(
        "usage: java ConstituencyParse -tokenize - -tokpath <tokpath> -parentpath <parentpath>");
      System.exit(1);
    }

    boolean tokenize = false;
    if (props.containsKey("tokenize")) {
      tokenize = true;
    }

    String tokPath = props.getProperty("tokpath");
    String parentPath = props.getProperty("parentpath");

    BufferedWriter tokWriter = new BufferedWriter(new FileWriter(tokPath));
    BufferedWriter parentWriter = new BufferedWriter(new FileWriter(parentPath));

    LexicalizedParser parser = LexicalizedParser.loadModel(
      "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
    TreeBinarizer binarizer = TreeBinarizer.simpleTreeBinarizer(
      parser.getTLPParams().headFinder(), parser.treebankLanguagePack());
    CollapseUnaryTransformer transformer = new CollapseUnaryTransformer();

    Scanner stdin = new Scanner(System.in);
    int count = 0;
    long start = System.currentTimeMillis();
    while (stdin.hasNextLine()) {
      String line = stdin.nextLine();
      List<HasWord> tokens = new ArrayList<>();
      if (tokenize) {
        PTBTokenizer<Word> tokenizer = new PTBTokenizer(
          new StringReader(line), new WordTokenFactory(), "");
        for (Word label; tokenizer.hasNext(); ) {
          tokens.add(tokenizer.next());
        }
      } else {
        for (String word : line.split(" ")) {
          tokens.add(new Word(word));
        }
      }

      Tree tree = parser.apply(tokens);
      Tree binarized = binarizer.transformTree(tree);
      Tree collapsedUnary = transformer.transformTree(binarized);
      Trees.convertToCoreLabels(collapsedUnary);
      collapsedUnary.indexSpans();

      List<Tree> leaves = collapsedUnary.getLeaves();
      int size = collapsedUnary.size() - leaves.size();
      int[] parents = new int[size];
      HashMap<Integer, Integer> index = new HashMap<Integer, Integer>();

      int idx = leaves.size();
      int leafIdx = 0;
      for (Tree leaf : leaves) {
        Tree cur = leaf.parent(collapsedUnary); // go to preterminal
        int curIdx = leafIdx++;
        boolean done = false;
        while (!done) {
          Tree parent = cur.parent(collapsedUnary);
          if (parent == null) {
            parents[curIdx] = 0;
            break;
          }

          int parentIdx;
          int parentNumber = parent.nodeNumber(collapsedUnary);
          if (!index.containsKey(parentNumber)) {
            parentIdx = idx++;
            index.put(parentNumber, parentIdx);
          } else {
            parentIdx = index.get(parentNumber);
            done = true;
          }

          parents[curIdx] = parentIdx + 1;
          cur = parent;
          curIdx = parentIdx;
        }
      }

      // print tokens
      int len = tokens.size();
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < len - 1; i++) {
        if (tokenize) {
          sb.append(PTBTokenizer.ptbToken2Text(tokens.get(i).word()));
        } else {
          sb.append(tokens.get(i).word());
        }
        sb.append(' ');
      }
      if (tokenize) {
        sb.append(PTBTokenizer.ptbToken2Text(tokens.get(len - 1).word()));
      } else {
        sb.append(tokens.get(len - 1).word());
      }
      sb.append('\n');
      tokWriter.write(sb.toString());

      // print parent pointers
      sb = new StringBuilder();
      for (int i = 0; i < size - 1; i++) {
        sb.append(parents[i]);
        sb.append(' ');
      }
      sb.append(parents[size - 1]);
      sb.append('\n');
      parentWriter.write(sb.toString());

      count++;
      if (count % 1000 == 0) {
        double elapsed = (System.currentTimeMillis() - start) / 1000.0;
        System.err.printf("Parsed %d lines (%.2fs)\n", count, elapsed);
      }
    }

    long totalTimeMillis = System.currentTimeMillis() - start;
    System.err.printf("Done: %d lines in %.2fs (%.1fms per line)\n",
      count, totalTimeMillis / 1000.0, totalTimeMillis / (double) count);
    tokWriter.close();
    parentWriter.close();
  }
}
