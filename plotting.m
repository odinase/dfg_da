error_names = ["abs_errors"];%, "misdetection_errors"];%, "detection_errors", "nonexistence_errors", "max_errors", "raw_errors"];
lbp_path = "./pmbm_analysis_output copy/lbp/errors";
williams_path = "./pmbm_analysis_output copy/williams/errors";
method_names = ["lbp", "williams"];

lbp_errors = errors_from_path(lbp_path, error_names);
williams_errors = errors_from_path(williams_path, error_names);

plot_survival_function({lbp_errors, williams_errors}, error_names, method_names);

function errors = errors_from_path(path, error_names)

errors = containers.Map;

for error_name = error_names
    f = fopen(path + "/" + error_name + ".bin");
    error = fread(f,inf,'*float64',0,'l');
    fclose(f);
    
    errors(error_name) = error;

end

end

function plot_survival_function(method_errors, error_names, method_names)

nrows = length(error_names);
n_methods = length(method_errors);

for i = 1:nrows
    subplot(nrows, 1, i);
    error_name = error_names(i);

    for j = 1:n_methods

        error = sort(method_errors{j}(error_name));
        z = find(error > 0, 1, 'first');
        error(z:end) = log10(error(z:end));
        steps = linspace(1.0, 0.0, length(error));
        stairs(error, steps, 'DisplayName', method_names(j));
        hold on

    end
    hold off

    title(strrep(error_names(i), '_', ' '));
    gca.XMinorTick = 'on';
    gca.XAxis.TickValues = linspace(min(error), max(error), length(error));
    gca.XAxis.Exponent = 1;
    %symlog(gca,'x', 1e-20);
    set(gca, 'YScale','log');
    %set(gca,'XScale','log', 'YScale','log');
    legend();

end

end