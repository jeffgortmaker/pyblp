% This script performs regressions and generates scatter plots

if ~isdeployed
  main();
end

function main()
  mpg_clean = readtable('../input/mpg.csv');
  regression_table(mpg_clean);
  city_figure(mpg_clean);
  hwy_figure(mpg_clean);
end

function regression_table(data)
  % OLS regression: displacement ~ city fuel economy
  mdl_cty = fitlm(data, 'displ ~ cty');
  disp(mdl_cty);
  
  % OLS regression: displacement ~ highway fuel economy
  mdl_hwy = fitlm(data, 'displ ~ hwy');
  disp(mdl_hwy);
  
  % OLS regression: displacement ~ city + highway fuel economy
  mdl_hwy_cty = fitlm(data, 'displ ~ cty + hwy');
  disp(mdl_hwy_cty);
  
  % Write the regression summary as LaTeX
  latex_table = extract_latex_table(mdl_hwy_cty);
  fid = fopen('../output/table_reg.tex', 'w');
  fprintf(fid, '%s', latex_table);
  fclose(fid);
end

% Utility function to extract LaTeX table
function latex_table = extract_latex_table(mdl)
  % Extract coefficients, standard errors, t-stats, and p-values
  coef = mdl.Coefficients.Estimate;
  se = mdl.Coefficients.SE;
  tstat = mdl.Coefficients.tStat;
  pvalue = mdl.Coefficients.pValue;
  
  % Start building the LaTeX table 
  latex_table = sprintf('\\begin{tabular}{lcccc}\n');
  latex_table = strcat(latex_table, sprintf('Variable & Estimate & SE & tStat & pValue \\\\ \\hline\n'));

  % Loop over coefficients and add them to the table
  for i = 1:length(coef)
      line = sprintf('%s & %.4f & %.4f & %.4f & %.4g \\\\ \n', ...
          mdl.CoefficientNames{i}, coef(i), se(i), tstat(i), pvalue(i));
      latex_table = strcat(latex_table, line);
  end

  % Close the LaTeX table
  latex_table = strcat(latex_table, sprintf('\\end{tabular}\n'));
end


function hwy_figure(data)
  scatter(data.displ, data.hwy, 50, data.year, 'filled');
  xlabel('Engine displacement (L)');
  ylabel('Highway fuel economy (mpg)');
  saveas(gcf, '../output/figure_hwy.jpg');
end

function city_figure(data)
  scatter(data.displ, data.cty, 50, data.year, 'filled');
  xlabel('Engine displacement (L)');
  ylabel('City fuel economy (mpg)');
  saveas(gcf, '../output/figure_city.jpg');
end

