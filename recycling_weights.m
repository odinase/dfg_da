function [p_P] = recycling_weights(probLogLocal, hyposLocal, LBP_marginals, bethe_likelihood, trackNumberLookup)

num_tracks = size(LBP_marginals, 2);
num_measurements = size(LBP_marginals, 2) - 2;
murty_marginals = zeros(size(LBP_marginals));

begsH = tCloud2BegInd(hyposCardLocal);
endsH = tCloud2EndInd(hyposCardLocal);

nH = length(hyposCardLocal);

Z_murty = exp(logsumexp(probLogLocal, 2));
Z_bethe = exp(bethe_likelihood);
Z_P = Z_bethe - Z_murty;


for iH = 1:nH
    tracks_not_in_hypo = 1:num_tracks;
    tracks_in_hypo = hyposLocal(begsH(iH):endsH(iH));
    old_tracks = map_new_tracks_to_old_tracks(tracks_in_hypo, trackNumberLookup);
    tracks_not_in_hypo(old_tracks) = [];
    hypo_prob = exp(probLogLocal(iH) - bethe_likelihood);
    for track = tracks_in_hypo
        [t,j] = find(trackNumberLookup == track);
        % If the track is made from misdetection, map it to 1
        if j > num_measurements
            j = 1;
        else
            % We need to offset by 1 if it was detection
            j = j + 1;
        end
        murty_marginals(j, t) = murty_marginals(j, t) + hypo_prob;
    end
    for track = tracks_not_in_hypo
        murty_marginals(num_measurements + 2, track) = murty_marginals(num_measurements + 2, track) + hypo_prob;
    end
end

% Now that we have the Murty marginals, compute the recycling weights
p_P = Z_bethe / Z_P * (LBP_marginals - murty_marginals);


end



function original_tracks = map_new_tracks_to_old_tracks(new_tracks, trackNumberLookup)
    original_tracks = zeros(1, length(new_tracks));
    k = 1;
    for new_track = new_tracks
        [t, ~] = find(trackNumberLookup == new_track);
        original_tracks(k) = t;
        k = k + 1;
    end

    original_tracks = unique(original_tracks);
end