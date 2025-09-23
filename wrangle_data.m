% Short description of script's purpose

if ~isdeployed
    main();
end

function main()
    mpg = readtable('../input/mpg.csv', 'VariableNamingRule', 'preserve');
    mpg_clean = clean_mpg_data(mpg);
    writetable(mpg_clean, '../output/mpg.csv');
end

function mpg_clean = clean_mpg_data(mpg)
    % Data wrangling steps here
    mpg_clean = mpg;
end
