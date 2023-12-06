import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))



META_TASKS = {
            'cramped_room': ['place_onion_in_pot', 'deliver_soup', 'place_onion_and_deliver_soup', 'random'],
            # 'cramped_room': ['place_onion_in_pot', 'deliver_soup', 'random'],

            # 'marshmallow_experiment': ['place_onion_in_pot', 'place_tomato_in_pot', 'deliver_soup', 'random', 'place_onion_and_deliver_soup', 'place_tomato_and_deliver_soup'],
             'marshmallow_experiment': ['place_onion_in_pot', 'place_tomato_in_pot', 'deliver_soup', 'random'],

            'asymmetric_advantages': ['place_onion_in_pot', 'deliver_soup', 'place_onion_and_deliver_soup', 'random'],
            # 'asymmetric_advantages': ['place_onion_in_pot', 'deliver_soup'],

            'coordination_ring': ['place_onion_in_pot', 'deliver_soup', 'place_onion_and_deliver_soup', 'random'],
            # 'coordination_ring': ['place_onion_in_pot', 'deliver_soup']

}

FCP_MODELS = {'cramped_room': os.path.join(_curr_directory, 'fcp/fcp_cramped_room_seed1.pt'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'fcp/fcp_marshmallow_experiment_seed1.pt'),
              'asymmetric_advantages': os.path.join(_curr_directory, 'fcp/fcp_asymmetric_advantages_seed1.pt'),
              'coordination_ring': os.path.join(_curr_directory, 'fcp/fcp_coordination_ring_seed1.pt')
              }

SP_MODELS = {'cramped_room': os.path.join(_curr_directory, 'sp/cramped_room_sp_periodic_3000.pt'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'sp/marshmallow_experiment_sp_periodic_3000.pt'),
              'asymmetric_advantages': os.path.join(_curr_directory, 'sp/asymmetric_advantages_sp_periodic_3000.pt'),
              'coordination_ring': os.path.join(_curr_directory, 'sp/coordination_ring_sp_periodic_3000.pt')
              }

BCP_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bcp/bcp_cramped_room-seed1.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bcp/bcp_marshmallow_experiment-seed1.pth'),
              'asymmetric_advantages': os.path.join(_curr_directory, 'bcp/bcp_asymmetric_advantages-seed1.pth'),
              'coordination_ring': os.path.join(_curr_directory, 'bcp/bcp_coordination_ring-seed1.pth')
              }

BC_MODELS = {'cramped_room': os.path.join(_curr_directory, 'bc/BC_cramped_room.pth'),
             'marshmallow_experiment': os.path.join(_curr_directory, 'bc/BC_marshmallow_experiment.pth'),
             'asymmetric_advantages': os.path.join(_curr_directory, 'bc/BC_asymmetric_advantages.pth'),
             'coordination_ring': os.path.join(_curr_directory, 'bc/BC_coordination_ring.pth'),
             }

SKILL_MODELS = {
    'cramped_room': [os.path.join(_curr_directory, 'policy_pool_alt_agent/cramped_room/fcp/s1/sp1_init_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/cramped_room/fcp/s1/sp1_mid_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/cramped_room/fcp/s1/sp1_final_actor.pt')],
             'marshmallow_experiment': [os.path.join(_curr_directory, 'policy_pool_alt_agent/marshmallow_experiment/fcp/s1/sp1_init_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/marshmallow_experiment/fcp/s1/sp1_mid_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/marshmallow_experiment/fcp/s1/sp1_final_actor.pt')],
              'asymmetric_advantages': [os.path.join(_curr_directory, 'policy_pool_alt_agent/asymmetric_advantages/fcp/s1/sp1_init_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/asymmetric_advantages/fcp/s1/sp1_mid_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/asymmetric_advantages/fcp/s1/sp1_final_actor.pt')],
              'coordination_ring': [os.path.join(_curr_directory, 'policy_pool_alt_agent/coordination_ring/fcp/s1/sp1_init_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/coordination_ring/fcp/s1/sp1_mid_actor.pt'),
                                 os.path.join(_curr_directory, 'policy_pool_alt_agent/coordination_ring/fcp/s1/sp1_final_actor.pt')]
              }

NN_MODELS = {'cramped_room':
             # [os.path.join(_curr_directory, 'NN/NN_cramped_room_place_onion_in_pot_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_cramped_room_deliver_soup_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_cramped_room_place_onion_and_deliver_soup_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_cramped_room_random_s_prime_r.pth')],
             [os.path.join(_curr_directory, 'RNN/rnn_cramped_room_place_onion_in_pot_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_cramped_room_deliver_soup_s_prime_r.pth'),
              # os.path.join(_curr_directory, 'RNN/rnn_cramped_room_place_onion_and_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_cramped_room_random_s_prime_r.pth')
              ],

            'marshmallow_experiment':
                # [os.path.join(_curr_directory,
                #                                     'NN/NN_marshmallow_experiment_place_onion_in_pot_s_prime_r.pth'),
                #                        os.path.join(_curr_directory,
                #                                     'NN/NN_marshmallow_experiment_place_tomato_in_pot_s_prime_r.pth'),
                #                        os.path.join(_curr_directory,
                #                                     'NN/NN_marshmallow_experiment_deliver_soup_s_prime_r.pth'),
                #                         os.path.join(_curr_directory,
                #                                      'NN/NN_marshmallow_experiment_random_s_prime_r.pth'),
                #                         os.path.join(_curr_directory,
                #                                      'NN/NN_marshmallow_experiment_place_onion_and_deliver_soup_s_prime_r.pth'),
                #                        os.path.join(_curr_directory,
                #                                     'NN/NN_marshmallow_experiment_place_tomato_and_deliver_soup_s_prime_r.pth'),
                #                        ],
                                        [os.path.join(_curr_directory,
                                                    'RNN/rnn_marshmallow_experiment_place_onion_in_pot_s_prime_r.pth'),
                                       os.path.join(_curr_directory,
                                                    'RNN/rnn_marshmallow_experiment_place_tomato_in_pot_s_prime_r.pth'),
                                       os.path.join(_curr_directory,
                                                    'RNN/rnn_marshmallow_experiment_deliver_soup_s_prime_r.pth'),
                                        os.path.join(_curr_directory,
                                                     'RNN/rnn_marshmallow_experiment_random_s_prime_r.pth'),
                                       #  os.path.join(_curr_directory,
                                       #               'RNN/rnn_marshmallow_experiment_place_onion_and_deliver_soup_s_prime_r.pth'),
                                       # os.path.join(_curr_directory,
                                       #              'RNN/rnn_marshmallow_experiment_place_tomato_and_deliver_soup_s_prime_r.pth'),
                                       ],
            'asymmetric_advantages':
             # [os.path.join(_curr_directory, 'NN/NN_asymmetric_advantages_place_onion_in_pot_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_asymmetric_advantages_deliver_soup_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_asymmetric_advantages_place_onion_and_deliver_soup_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_asymmetric_advantages_random_s_prime_r.pth')],
             [os.path.join(_curr_directory, 'RNN/rnn_asymmetric_advantages_place_onion_in_pot_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_asymmetric_advantages_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_asymmetric_advantages_place_onion_and_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_asymmetric_advantages_random_s_prime_r.pth')
              ],

            'coordination_ring':
             # [os.path.join(_curr_directory, 'NN/NN_coordination_ring_place_onion_in_pot_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_coordination_ring_deliver_soup_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_coordination_ring_place_onion_and_deliver_soup_s_prime_r.pth'),
             #  os.path.join(_curr_directory, 'NN/NN_coordination_ring_random_s_prime_r.pth')]
            [os.path.join(_curr_directory, 'RNN/rnn_coordination_ring_place_onion_in_pot_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_coordination_ring_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_coordination_ring_place_onion_and_deliver_soup_s_prime_r.pth'),
              os.path.join(_curr_directory, 'RNN/rnn_coordination_ring_random_s_prime_r.pth')
             ]
             }

MTP_MODELS = {'cramped_room':
              [os.path.join(_curr_directory,
                            'mtp/cramped_room/mtp_cramped_room-place_onion_in_pot-seed1.pth'),
                           os.path.join(_curr_directory,
                                        'mtp/cramped_room/mtp_cramped_room-deliver_soup-seed1.pth'),
                           os.path.join(_curr_directory,
                                        'mtp/cramped_room/mtp_cramped_room-place_onion_and_deliver_soup-seed1.pth'),
                           os.path.join(_curr_directory,
                                        'mtp/cramped_room/mtp_cramped_room-random-seed1.pth'),
                           ],

              'marshmallow_experiment':
                  [os.path.join(_curr_directory,
                                'mtp/marshmallow_experiment/mtp_marshmallow_experiment-place_onion_in_pot-seed1.pth'),
                  os.path.join(_curr_directory,
                               'mtp/marshmallow_experiment/mtp_marshmallow_experiment-place_tomato_in_pot-seed1.pth'),
                  os.path.join(_curr_directory,
                               'mtp/marshmallow_experiment/mtp_marshmallow_experiment-deliver_soup-seed1.pth'),
                  os.path.join(_curr_directory,
                               'mtp/marshmallow_experiment/mtp_marshmallow_experiment-random-seed1.pth'),
                  # os.path.join(_curr_directory,
                  #              'mtp/marshmallow_experiment/mtp_marshmallow_experiment-place_onion_and_deliver_soup-seed1.pth'),
                  # os.path.join(_curr_directory,
                  #              'mtp/marshmallow_experiment/mtp_marshmallow_experiment-place_tomato_and_deliver_soup-seed1.pth')
                  ],

              'asymmetric_advantages':
                  [os.path.join(_curr_directory,
                                'mtp/asymmetric_advantages/mtp_asymmetric_advantages-place_onion_in_pot-seed1.pth'),
                   os.path.join(_curr_directory,
                                'mtp/asymmetric_advantages/mtp_asymmetric_advantages-deliver_soup-seed1.pth'),
                   os.path.join(_curr_directory,
                                'mtp/asymmetric_advantages/mtp_asymmetric_advantages-place_onion_and_deliver_soup-seed1.pth'),
                   os.path.join(_curr_directory,
                                'mtp/asymmetric_advantages/mtp_asymmetric_advantages-random-seed1.pth'),
                   ],
                'coordination_ring':
                    [os.path.join(_curr_directory,
                            'mtp/coordination_ring/mtp_coordination_ring-place_onion_in_pot-seed1.pth'),
                           os.path.join(_curr_directory,
                                        'mtp/coordination_ring/mtp_coordination_ring-deliver_soup-seed1.pth'),
                           os.path.join(_curr_directory,
                                        'mtp/coordination_ring/mtp_coordination_ring-place_onion_and_deliver_soup-seed1.pth'),
                           os.path.join(_curr_directory,
                                        'mtp/coordination_ring/mtp_coordination_ring-random-seed1.pth'),
                           ],
                  }

METATASK_MODELS = {'cramped_room':
              [os.path.join(_curr_directory,'opponent/opponent_cramped_room_place_onion_in_pot.pth'),
                           os.path.join(_curr_directory,'opponent/opponent_cramped_room_deliver_soup.pth'),
                           os.path.join(_curr_directory,'opponent/opponent_cramped_room_place_onion_and_deliver_soup.pth'),
                           os.path.join(_curr_directory,'opponent/opponent_cramped_room_random.pth'),
                           ],

              'marshmallow_experiment':
                  [os.path.join(_curr_directory,
                                'opponent/opponent_marshmallow_experiment_place_onion_in_pot.pth'),
                  os.path.join(_curr_directory,
                               'opponent/opponent_marshmallow_experiment_place_tomato_in_pot.pth'),
                  os.path.join(_curr_directory,
                               'opponent/opponent_marshmallow_experiment_deliver_soup.pth'),
                  os.path.join(_curr_directory,
                               'opponent/opponent_marshmallow_experiment_random.pth'),
                  # os.path.join(_curr_directory,
                  #              'opponent/opponent_marshmallow_experiment_place_onion_and_deliver_soup.pth'),
                  # os.path.join(_curr_directory,
                  #              'opponent/opponent_marshmallow_experiment_place_tomato_and_deliver_soup.pth')
                  ],

              'asymmetric_advantages':
                  [os.path.join(_curr_directory,
                                'opponent/opponent_asymmetric_advantages_place_onion_in_pot.pth'),
                   os.path.join(_curr_directory,
                                'opponent/opponent_asymmetric_advantages_deliver_soup.pth'),
                   os.path.join(_curr_directory,
                                'opponent/opponent_asymmetric_advantages_place_onion_and_deliver_soup.pth'),
                   os.path.join(_curr_directory,
                                'opponent/opponent_asymmetric_advantages_random.pth'),
                   ],
                'coordination_ring':
                       [os.path.join(_curr_directory,
                                  'opponent/opponent_coordination_ring_place_onion_in_pot.pth'),
                           os.path.join(_curr_directory,
                                        'opponent/opponent_coordination_ring_deliver_soup.pth'),
                           os.path.join(_curr_directory,
                                        'opponent/opponent_coordination_ring_place_onion_and_deliver_soup.pth'),
                           os.path.join(_curr_directory,
                                        'opponent/opponent_coordination_ring_random.pth'),
                           ],
                  }




