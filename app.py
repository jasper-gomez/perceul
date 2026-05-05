import gradio as gr

from perceul.numeric_selector import NumericSelector
from perceul.clustering import explore_clusters, final_clustering

#========== GRADIO INTERFACE ==========
with gr.Blocks(title="PERCEUL: Perception-Based Worker Profiler") as app:

    gr.Markdown("# 🧠 PERCEUL: Profiler of Perception and Cognitive Ergonomics in the Workplace")

    file_input = gr.File(label="Upload CSV")
    with gr.Tab("Cluster Exploration"):
        
        btn = gr.Button("Explore Clusters")
        
        plot_output = gr.Plot()
        cluster_summary_output = gr.JSON(label="Cluster Sizes")
        outlier_output = gr.Markdown("### Exploration Report")
    
    with gr.Tab("Final Clustering"):
        top_features = gr.Number(
            value=5, 
            label="Number of Features to Display", 
            minimum=3, 
            maximum=10,
            step=1,
            precision=0
        )
    
        run_btn = gr.Button("Run Final Clustering")
        best_k_out = gr.Number(label="Number of Clusters `K` (Read-Only)", interactive=False, precision=0)
        gr.Markdown("### Cluster Characteristics")
        deviations_out = gr.Markdown()
    
    # ==========================================================
    #                   BUTTON CALLBACKS
    # ==========================================================
    
    run_btn.click(
        final_clustering,
        inputs=[file_input, top_features],
        outputs=[best_k_out, deviations_out]
    )

    btn.click(
            fn=explore_clusters,
            inputs=[file_input],
            outputs=[plot_output, cluster_summary_output, outlier_output]
        )

app.launch()