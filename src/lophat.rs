use itertools::Itertools;
use lophat::columns::{Column, VecColumn};
use spade::Triangulation;
use std::collections::HashMap;

use crate::DelaunayFiltration;

/// Extracts the sparse boundary matrix of the filtration of the Delaunay triangulation.
/// Additionally returns the times at which each column enters the filtration.
/// Columns are sorted by dimension and then filtration time.
pub fn sparsify(filtration: &DelaunayFiltration) -> (Vec<VecColumn>, Vec<f64>) {
    let mut vertex_idxs = HashMap::new();
    let mut edge_idxs = HashMap::new();
    let mut next_idx = 0;
    let mut cols = vec![];
    let mut filtration_times = vec![];

    // Add vertices according to filtration
    let vertex_list: Vec<_> = filtration
        .fixed_vertices()
        .map(|vertex| (vertex, filtration.vertex(vertex).data().filtration))
        .sorted_by(|pair_a, pair_b| pair_a.1.partial_cmp(&pair_b.1).unwrap())
        .collect();
    for (vertex, vertex_time) in vertex_list {
        // Add sparse column
        cols.push(VecColumn::new_with_dimension(0));
        // Record index of this vertex
        vertex_idxs.insert(vertex, next_idx);
        // Increment index
        next_idx += 1;
        // Record filtration
        filtration_times.push(vertex_time);
    }

    // Add edges according to filtration
    let edge_list: Vec<_> = filtration
        .fixed_undirected_edges()
        .map(|edge| (edge, filtration.undirected_edge(edge).data().filtration))
        .sorted_by(|pair_a, pair_b| pair_a.1.partial_cmp(&pair_b.1).unwrap())
        .collect();
    for (edge, edge_time) in edge_list {
        // Add sparse column
        let vertices = filtration.undirected_edge(edge).vertices();
        let boundary: Vec<_> = vertices
            .into_iter()
            .map(|v| {
                *vertex_idxs
                    .get(&v.fix())
                    .expect("Should have already added vertices")
            })
            .sorted()
            .collect();
        let edge_col = VecColumn::from((1, boundary));
        cols.push(edge_col);
        // Record index of this edge
        edge_idxs.insert(edge, next_idx);
        // Increment index
        next_idx += 1;
        // Record filtration
        filtration_times.push(edge_time);
    }

    // Add faces according to filtration
    let face_list: Vec<_> = filtration
        .fixed_inner_faces()
        .map(|face| (face, filtration.face(face).data().filtration))
        .sorted_by(|pair_a, pair_b| pair_a.1.partial_cmp(&pair_b.1).unwrap())
        .collect();
    for (face, face_time) in face_list {
        // Add sparse column
        let edges = filtration.face(face).adjacent_edges();
        let boundary: Vec<_> = edges
            .into_iter()
            .map(|edge| {
                *edge_idxs
                    .get(&edge.as_undirected().fix())
                    .expect("Should have already added vertices")
            })
            .sorted()
            .collect();
        let face_col = VecColumn::from((2, boundary));
        cols.push(face_col);
        // Increment index
        next_idx += 1;
        // Record filtration
        filtration_times.push(face_time);
    }

    (cols, filtration_times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{alpha_filtration, tests::simple_square, weak_alpha_filtration};
    use itertools::Itertools;
    use lophat::{
        algorithms::{RVDecomposition, SerialAlgorithm},
        utils::PersistenceDiagram,
    };
    use spade::Point2;

    fn f_time_diagram(
        mut diagram: PersistenceDiagram,
        f_times: &Vec<f64>,
    ) -> (Vec<(f64, f64)>, Vec<f64>) {
        let paired: Vec<_> = diagram
            .paired
            .drain()
            .map(|(b_idx, d_idx)| (f_times[b_idx], f_times[d_idx]))
            .filter(|&(b, d)| b != d)
            .sorted_by(|(b1, _d1), (b2, _d2)| b1.partial_cmp(b2).unwrap())
            .collect();
        let unpaired: Vec<_> = diagram
            .unpaired
            .drain()
            .map(|b_idx| f_times[b_idx])
            .sorted_by(|b1, b2| b1.partial_cmp(b2).unwrap())
            .collect();
        (paired, unpaired)
    }

    #[test]
    fn alpha_persistence() {
        let points = simple_square();
        let filtration = alpha_filtration(points);
        let (matrix, f_times) = sparsify(&filtration);
        let diagram = SerialAlgorithm::decompose(matrix.into_iter(), None).diagram();
        let (paired, unpaired) = f_time_diagram(diagram, &f_times);
        let correct_paired = vec![
            // Three 0-dimensional
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            // One 1-dimension
            (1.0, 2.0),
        ];
        // A single infinite 0-dimensional feature
        let correct_unpaired = vec![0.0];
        assert_eq!(paired, correct_paired);
        assert_eq!(unpaired, correct_unpaired);
    }

    #[test]
    fn weak_alpha_persistence() {
        let points = simple_square();
        let filtration = weak_alpha_filtration(points);
        let (matrix, f_times) = sparsify(&filtration);
        let diagram = SerialAlgorithm::decompose(matrix.into_iter(), None).diagram();
        let (paired, unpaired) = f_time_diagram(diagram, &f_times);
        let correct_paired = vec![
            // Three 0-dimensional
            (0.0, 4.0),
            (0.0, 4.0),
            (0.0, 4.0),
            // One 1-dimension
            (4.0, 8.0),
        ];
        // A single infinite 0-dimensional feature
        let correct_unpaired = vec![0.0];
        assert_eq!(paired, correct_paired);
        assert_eq!(unpaired, correct_unpaired);
    }

    // The point of this test is that the circum center is outside of the triangle
    #[test]
    fn alpha_wide_isoceles() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 1.0),
        ];
        let filtration = alpha_filtration(points);
        let (matrix, f_times) = sparsify(&filtration);
        let diagram = SerialAlgorithm::decompose(matrix.into_iter(), None).diagram();
        let (paired, unpaired) = f_time_diagram(diagram, &f_times);
        let correct_paired = vec![
            // Two 0-dimensional
            (0.0, 6.5),
            (0.0, 6.5),
            // No 1-dimensional
        ];
        // A single infinite 0-dimensional feature
        let correct_unpaired = vec![0.0];
        assert_eq!(paired, correct_paired);
        assert_eq!(unpaired, correct_unpaired);
    }
}
